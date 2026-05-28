from pathlib import Path

import cv2
import streamlit as st
from streamlit_autorefresh import st_autorefresh

from drawing import draw_predictions
from predict import run_inference
from preprocess import preprocess
from smoothing import PredictionSmoother
from utils.image_utils import resize_for_inference, pil_to_bgr, bgr_to_rgb
import time
import threading

from PIL import Image
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
        "tts_cooldown_seconds": 5.0,
        "last_spoken_label": None,
        "last_spoken_time": 0.0,
        "is_speaking": False,
        "speaking_until": 0.0,
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)
    if "smoother" not in st.session_state:
        st.session_state["smoother"] = PredictionSmoother(buffer_size=5)


def update_results(label: str, confidence: float, detected: bool) -> None:
    st.session_state.detected_label = label
    st.session_state.detection_confidence = confidence
    detected_flag = confidence > 0.20
    st.session_state.has_detection = detected_flag
    if detected_flag:
        st.session_state.detection_count = st.session_state.get("detection_count", 0) + 1


def translate_label(label: str) -> str:
    translations = {
        "good-afternoon": "Magandang hapon (Good afternoon)",
        "good-evening":   "Magandang gabi (Good evening)",
        "good-morning":   "Magandang umaga (Good morning)",
        "goodbye":        "Paalam (Goodbye)",
        "hello":          "Kumusta (Hello)",
        "how-are-you":    "Kumusta ka? (How are you)",
        "i-love-you":     "Mahal kita (I love you)",
        "i'm-fine":       "Ayos lang ako (I'm fine)",
        "nice to meet you": "Ikinagagalak kong makilala ka (Nice to meet you)",
        "please":         "Pakiusap (Please)",
        "sorry":          "Paumanhin (Sorry)",
        "thank-you":      "Salamat (Thank you)",
        "what's-your-name": "Anong pangalan mo? (What's your name?)",
        "youre-welcome":  "Walang anuman (You're welcome)",
    }
    return translations.get(label, f"{label} (translation unavailable)")


def speak_filipino(text: str) -> None:
    """Synthesize speech in a background thread so the UI never blocks."""
    try:
        def _run():
            engine = pyttsx3.init()
            engine.setProperty("rate", 155)
            engine.say(text)
            engine.runAndWait()

        threading.Thread(target=_run, daemon=True).start()
        st.session_state["is_speaking"] = False
        st.session_state["speaking_until"] = time.time() + 1.5
        st.session_state["last_tts_text"] = text
    except Exception as e:
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

    display_text = translate_label(label) if label and label != "?" else label
    st.markdown(f'<div class="gesture-label">{display_text}</div>', unsafe_allow_html=True)

    if time.time() < float(st.session_state.get("speaking_until", 0.0)):
        st.caption("Speaking...")

    st.progress(min(max(confidence, 0.0), 1.0))
    st.caption(f"Confidence: {confidence:.0%}")

    last_text = st.session_state.get("last_tts_text", None)
    if last_text:
        if st.button("Replay voice"):
            speak_filipino(last_text)

    st.markdown('</div>', unsafe_allow_html=True)


# ─── Page setup ───────────────────────────────────────────────────────────────

st.set_page_config(page_title="FSL Detector", layout="wide")
load_css()
init_state()

# Auto-refresh every 1 second so the results panel stays in sync after capture
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
    st.caption("Point your hand at the camera and press the capture button to detect.")

    # Voice toggle
    st.checkbox(
        "Enable voice (Tagalog)",
        value=st.session_state.get("voice_enabled", True),
        key="voice_enabled",
    )

    # ── Camera input ──────────────────────────────────────────────────────────
    # st.camera_input opens the browser camera directly via HTML5.
    # No WebRTC peer connection, no STUN, no TURN required.
    # The user presses the shutter button; the captured frame is sent to
    # Streamlit as a standard image upload and processed by the CV pipeline.
    camera_frame = st.camera_input(
        label="camera",
        label_visibility="collapsed",
    )

    if camera_frame is not None:
        # Convert captured JPEG from browser to BGR numpy array for OpenCV
        pil_image = Image.open(camera_frame)
        bgr_image = pil_to_bgr(pil_image)
        bgr_resized = resize_for_inference(bgr_image)

        # Run the full CV pipeline: face mask → skin seg → morphology → contour
        annotated, hand_crop = preprocess(bgr_resized)

        # Only call Roboflow if a hand crop was successfully isolated
        if hand_crop is not None and hand_crop.size > 0:
            predictions = run_inference(hand_crop)
        else:
            predictions = []
            st.warning("No hand detected. Try better lighting or move your hand closer.")

        # Apply temporal smoothing across successive snapshots
        smoother = st.session_state["smoother"]

        if predictions:
            top = predictions[0]
            smoothed = smoother.smooth(top["class"], top["confidence"])
            update_results(smoothed, top["confidence"], True)

            # Speak detected sign if voice is enabled and cooldown has passed
            if st.session_state.get("voice_enabled", True):
                now = time.time()
                cooldown = float(st.session_state.get("tts_cooldown_seconds", 5.0))
                last_spoken_time = float(st.session_state.get("last_spoken_time", 0.0))
                if now - last_spoken_time >= cooldown:
                    filipino = translate_label(smoothed).split(" (")[0]
                    if filipino and filipino != "?":
                        speak_filipino(filipino)
                        st.session_state["last_spoken_label"] = smoothed
                        st.session_state["last_spoken_time"] = now
        else:
            update_results(smoother.stable_label, 0.0, False)

        # Draw bounding box and label overlay then display the annotated frame
        final = draw_predictions(annotated, predictions, smoother.stable_label)
        st.image(bgr_to_rgb(final), use_container_width=True)

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