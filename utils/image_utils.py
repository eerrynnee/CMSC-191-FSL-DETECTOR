import cv2
import numpy as np
from PIL import Image

# Target size for Roboflow model input 
ROBOFLOW_SIZE = 640


def resize_for_inference(frame):
    """
    Resize frame so the longest side equals ROBOFLOW_SIZE.
    Preserves aspect ratio, which is what the notebook does.

    Args:
        frame (np.ndarray): BGR frame.

    Returns:
        np.ndarray: Resized BGR frame.
    """
    height, width = frame.shape[:2]
    scale = ROBOFLOW_SIZE / max(height, width)
    new_w = round(scale * width)
    new_h = round(scale * height)
    return cv2.resize(frame, (new_w, new_h))


def bgr_to_rgb(frame):
    """
    Convert OpenCV BGR image to RGB for display in Streamlit.
    Streamlit's st.image() expects RGB, not BGR.

    Args:
        frame (np.ndarray): BGR image.

    Returns:
        np.ndarray: RGB image.
    """
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def pil_to_bgr(pil_image):
    """
    Convert a PIL Image (from st.file_uploader) to a BGR numpy array
    compatible with OpenCV functions.

    Args:
        pil_image (PIL.Image.Image): Image opened with PIL.

    Returns:
        np.ndarray: BGR image array.
    """
    rgb_array = np.array(pil_image.convert("RGB"))
    bgr_array = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2BGR)
    return bgr_array