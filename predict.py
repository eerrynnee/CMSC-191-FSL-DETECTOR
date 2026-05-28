from inference_sdk import InferenceHTTPClient

# Roboflow credentials — from the notebook
ROBOFLOW_API_KEY = "cKYxZ2Mmbixj3YYxwbeo"
ROBOFLOW_MODEL = "filipino-sign-language-recognition-mdkeq/1"

# Minimum confidence to include a prediction in results
MIN_CONFIDENCE = 0.40

# Initialize the Roboflow HTTP client once at module level
# This avoids re-creating the connection on every inference call
_client = InferenceHTTPClient(
    api_url="https://detect.roboflow.com",
    api_key=ROBOFLOW_API_KEY
)


def run_inference(image_bgr):
    """
    Send an image to the Roboflow model and return filtered predictions.

    The image should already be the hand crop from preprocess.py.
    Sending only the cropped hand region (not the full frame) reduces
    noise and improves model accuracy.

    Args:
        image_bgr (np.ndarray): BGR image to run inference on.

    Returns:
        list[dict]: List of prediction dicts with keys:
                    class, confidence, x, y, width, height.
                    Empty list if inference fails or no predictions pass threshold.
    """
    try:
        result = _client.infer(image_bgr, model_id=ROBOFLOW_MODEL)
        predictions = result.get("predictions", [])
        # Filter out low-confidence predictions
        filtered = [p for p in predictions if p["confidence"] > MIN_CONFIDENCE]
        return filtered
    except Exception as e:
        print(f"Inference error: {e}")
        return []