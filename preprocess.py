import cv2
import numpy as np

# Load Haar cascade once at module import time, not on every call
_face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# Padding added around the detected hand bounding box before cropping
HAND_CROP_PADDING = 30

# Minimum contour area to be considered a hand (filters out small noise blobs)
MIN_HAND_AREA = 2000


def preprocess(frame):
    """
    Full OpenCV preprocessing pipeline.

    Steps:
      1. Haar face masking  - blacks out face region so skin color detector ignores it
      2. HSV skin segmentation - isolates skin-colored pixels (hand region)
      3. Morphological cleanup - removes noise from the binary skin mask
      4. Contour extraction - finds the largest skin blob (assumed to be the hand)

    Args:
        frame (np.ndarray): BGR image from webcam or file upload.

    Returns:
        output_frame (np.ndarray): Copy of frame with face box and hand contour drawn.
        hand_crop (np.ndarray or None): Cropped hand region, or None if no hand found.
    """
    output_frame = frame.copy()
    working_frame = frame.copy()

    # Step 1: Detect and mask face region
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = _face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    for (fx, fy, fw, fh) in faces:
        # Black out face on the working frame to exclude skin color from face
        cv2.rectangle(working_frame,
                      (fx - 20, fy - 40), (fx + fw + 20, fy + fh + 40),
                      (0, 0, 0), -1)
        # Draw a visible red rectangle on output frame to show user the face was ignored
        cv2.rectangle(output_frame, (fx, fy), (fx + fw, fy + fh), (0, 0, 255), 2)
        cv2.putText(output_frame, "Face Ignored",
                    (fx, fy - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    # Step 2: HSV skin color segmentation
    # Two ranges cover reddish skin tones that wrap around the hue circle
    hsv = cv2.cvtColor(working_frame, cv2.COLOR_BGR2HSV)
    mask1 = cv2.inRange(hsv,
                        np.array([0,   20,  70],  dtype=np.uint8),
                        np.array([20,  255, 255], dtype=np.uint8))
    mask2 = cv2.inRange(hsv,
                        np.array([165, 20,  70],  dtype=np.uint8),
                        np.array([180, 255, 255], dtype=np.uint8))
    skin_mask = cv2.bitwise_or(mask1, mask2)

    # Step 3: Morphological operations to clean up mask noise
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_OPEN,  kernel, iterations=2)
    skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    # Step 4: Find contours and identify the largest (hand) region
    contours, _ = cv2.findContours(
        skin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    hand_crop = None

    if contours:
        largest = max(contours, key=cv2.contourArea)

        if cv2.contourArea(largest) > MIN_HAND_AREA:
            # Draw hand contour outline in green
            cv2.drawContours(output_frame, [largest], -1, (0, 255, 0), 2)

            # Compute bounding box with padding, clamped to frame edges
            x, y, w, h = cv2.boundingRect(largest)
            bx1 = max(0, x - HAND_CROP_PADDING)
            by1 = max(0, y - HAND_CROP_PADDING)
            bx2 = min(frame.shape[1], x + w + HAND_CROP_PADDING)
            by2 = min(frame.shape[0], y + h + HAND_CROP_PADDING)

            # Draw green bounding box on output frame
            cv2.rectangle(output_frame, (bx1, by1), (bx2, by2), (0, 255, 0), 3)

            # Crop the hand region from the original unmodified frame
            hand_crop = frame[by1:by2, bx1:bx2]

    return output_frame, hand_crop