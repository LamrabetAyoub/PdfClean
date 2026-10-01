#!/usr/bin/env python3
"""
job_queue.py — bound how much work runs at once, behind an interface.

The original API started a thread per upload. That is fine for one person
testing on a laptop and wrong the moment two people use it: ten uploads meant
ten documents rendering pages at 300 DPI simultaneously. Adding an LLM stage
makes it worse, because each job now also wants exclusive use of a model that
saturates the CPU on its own.

So work goes through a pool with a fixed number of document workers, and jobs
that arrive when every worker is busy are reported as `queued` rather than
started. The JobQueue interface exists so that swapping the in-process pool for
RQ or Celery later is a change to this file and the three lines in api.py that
construct it — not a change to the OCR or extraction services, which never
learn where they are running.

What this is not: durable. Jobs live in memory and die with the process.
See the deployment notes in README.md.
"""

import threading
from concurrent.futures import ThreadPoolExecutor


class JobQueue:
    """Interface a queue backend must satisfy."""

    def submit(self, job_id, fn, *args, **kwargs):
        """Schedule work. Must return immediately."""
        raise NotImplementedError

    def active_count(self):
        raise NotImplementedError

    def capacity(self):
        raise NotImplementedError

    def pending_count(self):
        raise NotImplementedError

    def shutdown(self, wait=False):
        raise NotImplementedError

    def would_queue(self):
        """True when a job submitted now would wait rather than start."""
        return self.active_count() >= self.capacity()


class ThreadPoolJobQueue(JobQueue):
    """In-process pool. The development and single-process default."""

    def __init__(self, workers=1):
        self._capacity = max(1, int(workers))
        self._pool = ThreadPoolExecutor(max_workers=self._capacity,
                                        thread_name_prefix="docworker")
        self._lock = threading.Lock()
        self._active = 0
        self._pending = 0

    def submit(self, job_id, fn, *args, **kwargs):
        with self._lock:
            self._pending += 1

        def run():
            with self._lock:
                self._pending -= 1
                self._active += 1
            try:
                return fn(*args, **kwargs)
            finally:
                with self._lock:
                    self._active -= 1

        return self._pool.submit(run)

    def active_count(self):
        with self._lock:
            return self._active

    def pending_count(self):
        with self._lock:
            return self._pending

    def capacity(self):
        return self._capacity

    def shutdown(self, wait=False):
        self._pool.shutdown(wait=wait)


class InlineJobQueue(JobQueue):
    """Runs work on the calling thread. For tests that want determinism."""

    def __init__(self):
        self._active = 0

    def submit(self, job_id, fn, *args, **kwargs):
        self._active += 1
        try:
            return fn(*args, **kwargs)
        finally:
            self._active -= 1

    def active_count(self):
        return self._active

    def pending_count(self):
        return 0

    def capacity(self):
        return 1

    def would_queue(self):
        return False

    def shutdown(self, wait=False):
        return None


def create(workers=1, inline=False):
    return InlineJobQueue() if inline else ThreadPoolJobQueue(workers)
