import threading
import queue
from typing import List

from predict import run_inference

# Single worker that consumes hand crops and updates latest predictions
_frame_q = queue.Queue(maxsize=2)
_latest_lock = threading.Lock()
_latest_predictions: List[dict] = []

_worker_thread = None
_stop_event = threading.Event()


def _worker():
    global _latest_predictions
    while not _stop_event.is_set():
        try:
            item = _frame_q.get(timeout=0.5)
        except Exception:
            continue

        hand_crop = item
        preds = []
        if hand_crop is not None:
            try:
                preds = run_inference(hand_crop)
            except Exception as e:
                print(f"Inference worker error: {e}")
                preds = []

        with _latest_lock:
            _latest_predictions = preds

        # mark task done if available
        try:
            _frame_q.task_done()
        except Exception:
            pass


def start_worker():
    global _worker_thread
    if _worker_thread is None or not _worker_thread.is_alive():
        _stop_event.clear()
        _worker_thread = threading.Thread(target=_worker, daemon=True)
        _worker_thread.start()


def stop_worker():
    _stop_event.set()
    # drain queue
    while not _frame_q.empty():
        try:
            _frame_q.get_nowait()
            _frame_q.task_done()
        except Exception:
            break


def submit_frame(hand_crop):
    # Non-blocking submit: replace oldest if full
    try:
        if _frame_q.full():
            try:
                _frame_q.get_nowait()
                _frame_q.task_done()
            except Exception:
                pass
        _frame_q.put_nowait(hand_crop)
    except Exception:
        pass


def get_latest_predictions():
    with _latest_lock:
        return list(_latest_predictions)
