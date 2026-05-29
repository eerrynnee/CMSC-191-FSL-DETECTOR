from pathlib import Path

import av
import cv2
import streamlit as st
from streamlit_webrtc import RTCConfiguration, VideoProcessorBase, webrtc_streamer
from streamlit_autorefresh import st_autorefresh

from drawing import draw_predictions
from predict import run_inference, MIN_CONFIDENCE
from inference_worker import start_worker, submit_frame, get_latest_predictions
from preprocess import preprocess
from smoothing import PredictionSmoother, CONF_THRESHOLD
from utils.image_utils import resize_for_inference
import tempfile
import os
import time
import threading

import pyttsx3


APP_DIR = Path(__file__).resolve().parent
CSS_PATH = APP_DIR / "assets" / "style.css"


def load_css() -> None:
    if CSS_PATH.exists():
        st.markdown(f"<style>{CSS_PATH.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)


def init_state() -> None:
    defaults = {
        "detected_label": "?",
        "detection_confidence": 0.0,
        "detection_count": 0,
        "has_detection": False,
        "voice_enabled": True,
        "tts_engine": "pyttsx3",
        "tts_cooldown_seconds": 5.0,
        "last_spoken_label": None,
        "last_spoken_time": 0.0,
        "is_speaking": False,
        "speaking_until": 0.0,
        "webcam_active": False,
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


def update_results(label: str, confidence: float, detected: bool) -> None:
    st.session_state.detected_label = label
    st.session_state.detection_confidence = confidence
    # Consider a detection active if confidence exceeds a reasonable threshold
    detected_flag = confidence > MIN_CONFIDENCE
    st.session_state.has_detection = detected_flag
    if detected_flag:
        st.session_state.detection_count = st.session_state.get("detection_count", 0) + 1


def translate_label(label: str) -> str:
    if label == "good-afternoon":
        return "Magandang hapon (Good afternoon)"
    elif label == "good-evening":
        return "Magandang gabi (Good evening)"
    elif label == "good-morning":
        return "Magandang umaga (Good morning)"
    elif label == "goodbye":
        return "Paalam (Goodbye)"
    elif label == "hello":
        return "Kumusta (hello)"
    elif label == "how-are-you":
        return "Kumusta ka? (How are you)"
    elif label == "i-love-you":
        return "Mahal kita (I love you)"
    elif label == "i-m fine":
        return "Ayos lang ako (I'm fine)"
    elif label == "nice to meet you":
        return "Ikinagagalak kong makilala ka (Nice to meet you)"
    elif label == "please":
        return "Pakiusap (Please)"
    elif label == "sorry":
        return "Paumanhin (Sorry)"
    elif label == "thank-you":
        return "Salamat (Thank you)"
    elif label == "what-s your name":
        return "Anong pangalan mo? (What's your name?)"
    elif label == "youre-welcome":
        return "Walang anuman (You're welcome)"
    else:
        return f"{label} (translation unavailable)"


def speak_filipino(text: str) -> None:
    """Synthesize Tagalog speech for `text` and play it via Streamlit audio."""
    try:
        def speak_with_pyttsx3() -> None:
            engine = pyttsx3.init()
            engine.setProperty("rate", 155)
            engine.say(text)
            engine.runAndWait()

        thread = threading.Thread(target=speak_with_pyttsx3, daemon=True)
        thread.start()
        st.session_state["is_speaking"] = False
        st.session_state["speaking_until"] = time.time() + 1.5
        st.session_state["last_tts_text"] = text
    except Exception as e:
        # Don't break the app if TTS fails; log to Streamlit
        st.write(f"TTS error: {e}")
        st.session_state["is_speaking"] = False


def render_results_panel() -> None:
    label = st.session_state.detected_label
    confidence = st.session_state.detection_confidence
    detected = st.session_state.has_detection

    
    status_html = (
        '<span class="status-active">Detection Active</span>'   
        if detected
        else '<span class="status-waiting">Waiting for hand</span>'
    )
    st.markdown(status_html, unsafe_allow_html=True)
    # Show label with confidence in the central results card
    # Translate dataset class into Filipino (English in parentheses) and display
    display_text = translate_label(label) if label and label != "?" else label
    st.markdown(f'<div class="gesture-label">{display_text}</div>', unsafe_allow_html=True)
    if st.session_state.get("is_speaking", False) or time.time() < float(st.session_state.get("speaking_until", 0.0)):
        st.caption("Speaking...")
    st.progress(min(max(confidence, 0.0), 1.0))
    st.caption(f"Confidence: {confidence:.0%}")
    last_text = st.session_state.get("last_tts_text", None)
    if st.session_state.get("webcam_active", False) and last_text:
        if st.button("Replay voice"):
            speak_filipino(last_text)
    st.markdown('</div>', unsafe_allow_html=True)


st.set_page_config(page_title="FSL Detector", layout="wide")
load_css()
init_state()

# Start background inference worker (threads safe; does nothing if already started)
start_worker()

# Auto-refresh the Streamlit page every second so we can read video-processor state
st_autorefresh(interval=1000, key="fsl_autorefresh")

st.markdown(
    """
    <div class="fsl-header">
        <h1>Filipino Sign Language Detector</h1>
        <p>Real-time hand gesture detection with live webcam input.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

left_col, right_col = st.columns([1.2, 0.8], gap="large")

with right_col:
    render_results_panel()

with left_col:
    st.subheader("Live webcam feed")
    st.caption("Allow camera access when prompted. Position your hand in frame.")
    # Voice toggle (Tagalog)
    st.checkbox("Enable voice (Tagalog)", value=st.session_state.get("voice_enabled", True), key="voice_enabled")

    # Keep ICE config empty to avoid external STUN retries that can crash on hosted environments.
    RTC_CONFIG = RTCConfiguration({"iceServers": []})

    class FSLVideoProcessor(VideoProcessorBase):
        def __init__(self):
            self.smoother = PredictionSmoother(buffer_size=5)
            self.result_label = "?"
            self.result_confidence = 0.0
            self.result_detection_count = 0
            # Raw immediate prediction (top of model output) — used for faster UI feedback
            self.raw_label = "?"
            self.raw_confidence = 0.0

        def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
            img = frame.to_ndarray(format="bgr24")
            img = cv2.flip(img, 1)
            img = resize_for_inference(img)
            annotated, hand_crop = preprocess(img)

            # Submit hand crop to background worker and fetch latest predictions
            if hand_crop is not None and hand_crop.size > 0:
                submit_frame(hand_crop)
            # Non-blocking fetch of latest available predictions
            predictions = get_latest_predictions()

            if predictions:
                top = predictions[0]
                # store raw top prediction for immediate UI display
                self.raw_label = top.get("class", "?")
                self.raw_confidence = float(top.get("confidence", 0.0))

                smoothed_label = self.smoother.smooth(self.raw_label, self.raw_confidence)
                self.result_label = smoothed_label
                self.result_confidence = self.raw_confidence
                self.result_detection_count += 1
            else:
                # no predictions: reset raw values and keep smoother stable label
                self.raw_label = "?"
                self.raw_confidence = 0.0
                self.result_label = self.smoother.stable_label

            annotated = draw_predictions(annotated, predictions, self.smoother.stable_label)
            return av.VideoFrame.from_ndarray(
                cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB),
                format="rgb24",
            )

    ctx = webrtc_streamer(
        key="fsl-webcam",
        video_processor_factory=FSLVideoProcessor,
        rtc_configuration=RTC_CONFIG,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )

    # Read processor state whenever available (playing check can be flaky)
    if getattr(ctx, "video_processor", None) is not None:
        processor = ctx.video_processor
        # Check webcam state before updating results so STOP clears the status immediately
        playing = None
        state = getattr(ctx, "state", None)
        if state is not None:
            playing = getattr(state, "playing", None)
        else:
            playing = getattr(ctx, "playing", None)

        if playing is False:
            st.session_state.webcam_active = False
            st.session_state.has_detection = False
            st.session_state.detected_label = "?"
            st.session_state.detection_confidence = 0.0
            st.session_state.last_spoken_label = None
            st.session_state.is_speaking = False
            st.session_state.speaking_until = 0.0
        else:
            st.session_state.webcam_active = True
        # Prefer the immediate raw prediction for UI feedback; fall back to smoothed label
            proc_raw_label = getattr(processor, "raw_label", "?")
            proc_raw_conf = getattr(processor, "raw_confidence", 0.0)
            proc_label = getattr(processor, "result_label", "?")
            proc_conf = getattr(processor, "result_confidence", 0.0)

            # Use raw prediction for display if it has reasonable confidence
            display_label = proc_raw_label if proc_raw_conf > CONF_THRESHOLD else proc_label
            display_conf = proc_raw_conf if proc_raw_conf > CONF_THRESHOLD else proc_conf

            update_results(display_label, display_conf, display_conf > MIN_CONFIDENCE)
            st.session_state.detection_count = getattr(processor, "result_detection_count", st.session_state.detection_count)

            # Speak every detected sign when confidence is above the detection threshold and voice is enabled
            if st.session_state.has_detection and display_label and display_label != "?":
                if st.session_state.get("voice_enabled", True):
                    now = time.time()
                    cooldown = float(st.session_state.get("tts_cooldown_seconds", 5.0))
                    last_spoken_time = float(st.session_state.get("last_spoken_time", 0.0))
                    if now - last_spoken_time >= cooldown:
                        filipino = translate_label(display_label).split(" (")[0]
                        if filipino and filipino != "?":
                            st.session_state["is_speaking"] = True
                            st.session_state["speaking_until"] = now + 1.5
                            speak_filipino(filipino)
                            st.session_state["is_speaking"] = False
                            st.session_state["last_spoken_label"] = display_label
                            st.session_state["last_spoken_time"] = now

st.markdown(
    """
    <div class="fsl-footer-bleed">
        <div class="fsl-footer">
            <div>
                <h4>CMSC 191 - Computer Vision Final Project</h4>
                <p>Filipino Sign Language Detection System</p>
            </div>
            <div>
                <h4>Team Members</h4>
                <p>Cabañero, Jomi Arielle</p>
                <p>Morgia, Miguel Alexis</p>
                <p>Torreon, Ericka Gwynne</p>
            </div>
            <div>
                <h4>Techniques Used</h4>
                <p>
                    Haar Cascade | HSV Segmentation | Morphological Ops | Contour Detection | YOLOv11 via Roboflow | Temporal Smoothing
                </p>
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

