import cv2

# Colors in BGR format
COLOR_LABEL_BG = (0, 0, 0)
COLOR_LABEL_TEXT = (0, 255, 255)   # Cyan text for detected label
COLOR_BBOX = (0, 255, 0)           # Green bounding box
COLOR_NO_DETECT_BG = (50, 50, 50)
COLOR_NO_DETECT_TEXT = (150, 150, 150)


def draw_predictions(frame, predictions, stable_label="?"):
    """
    Draw bounding box and label overlay on frame.

    When no predictions are present, shows the last stable label in gray
    to give the user continuous feedback even between detections.

    Args:
        frame (np.ndarray): BGR frame to draw on (will be modified in place).
        predictions (list[dict]): Prediction list from run_inference().
        stable_label (str): Last committed label from PredictionSmoother.

    Returns:
        np.ndarray: Annotated frame.
    """
    if not predictions:
        # No predictions: leave frame visuals minimal (no large header)
        return frame

    # Take only the top prediction
    pred = predictions[0]
    conf = pred["confidence"]
    label = pred["class"]

    # Draw bounding box
    # Roboflow returns center x, y and width, height — convert to corner coords
    x = int(pred["x"])
    y = int(pred["y"])
    w = int(pred["width"])
    h = int(pred["height"])
    x1, y1 = x - w // 2, y - h // 2
    x2, y2 = x + w // 2, y + h // 2
    cv2.rectangle(frame, (x1, y1), (x2, y2), COLOR_BBOX, 2)

    return frame