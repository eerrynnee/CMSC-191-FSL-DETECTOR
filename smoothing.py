from collections import deque

# Confidence threshold below which we ignore a prediction
CONF_THRESHOLD = 0.40 

# How many frames must agree before we commit to a new stable label
STABLE_THRESHOLD = 4


class PredictionSmoother:
    """
    Confidence-weighted temporal smoothing for sign predictions.

    Keeps a rolling buffer of recent (label, confidence) pairs.
    Uses weighted voting to reduce per-frame flickering.
    Only commits to a new label after STABLE_THRESHOLD consecutive agreements.
    """

    def __init__(self, buffer_size=7):
        self._buffer = deque(maxlen=buffer_size)
        self._stable_label = "?"
        self._stable_count = 0

    def smooth(self, label, confidence):
        """
        Accept a new prediction and return the current stable label.

        Args:
            label (str): Predicted class name from the model.
            confidence (float): Prediction confidence between 0 and 1.

        Returns:
            str: The smoothed, stable label.
        """
        # Ignore low-confidence predictions
        if confidence < CONF_THRESHOLD:
            return self._stable_label if self._stable_label != "?" else "?"

        self._buffer.append((label, confidence))

        # Weighted vote across buffer
        votes = {}
        for name, conf in self._buffer:
            votes[name] = votes.get(name, 0) + conf

        winner = max(votes, key=votes.get)

        # Only update stable label after enough consecutive frames agree
        if winner == self._stable_label:
            self._stable_count += 1
        else:
            self._stable_count = 1

        if self._stable_count >= STABLE_THRESHOLD:
            self._stable_label = winner

        return self._stable_label

    def reset(self):
        """Clear buffer and reset state. Call when switching input modes."""
        self._buffer.clear()
        self._stable_label = "?"
        self._stable_count = 0

    @property
    def stable_label(self):
        return self._stable_label