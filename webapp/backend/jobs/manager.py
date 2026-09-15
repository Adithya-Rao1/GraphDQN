import threading
from concurrent.futures import ThreadPoolExecutor

from webapp.backend.config import MAX_TRAINING_WORKERS


class JobManager:
    def __init__(self, max_workers: int = MAX_TRAINING_WORKERS):
        self._executor = ThreadPoolExecutor(max_workers=max_workers)
        self._cancel_flags: dict[int, threading.Event] = {}
        self._lock = threading.Lock()

    def _new_cancel_event(self, key: int) -> threading.Event:
        with self._lock:
            event = threading.Event()
            self._cancel_flags[key] = event
            return event

    def submit_training(self, run_id: int) -> None:
        from webapp.backend.jobs.training import run_training_job

        cancel_event = self._new_cancel_event(run_id)
        self._executor.submit(run_training_job, run_id, cancel_event)

    def submit_finetune(self, run_id: int) -> None:
        from webapp.backend.jobs.training import run_finetune_job

        cancel_event = self._new_cancel_event(run_id)
        self._executor.submit(run_finetune_job, run_id, cancel_event)

    def submit_generation(self, batch_id: int) -> None:
        from webapp.backend.jobs.generation import run_generation_job

        cancel_event = self._new_cancel_event(-batch_id)  # separate id space from training runs
        self._executor.submit(run_generation_job, batch_id, cancel_event)

    def cancel(self, run_id: int) -> bool:
        with self._lock:
            event = self._cancel_flags.get(run_id)
        if event is None:
            return False
        event.set()
        return True


job_manager = JobManager()
