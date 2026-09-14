"""Logging, progress reporting and cancellation shared by the CLI and the GUI."""

from __future__ import annotations

import threading
import time
from typing import Callable, Optional, Sequence, Tuple

LogCallback = Callable[[str], None]
ProgressCallback = Callable[[float, str], None]


class Cancelled(Exception):
    """Raised when the user cancels a running job."""


class Reporter:
    """Maps the progress of weighted stages onto a single 0..1 progress value.

    Usage::

        reporter.plan([("audio", 10), ("transcribe", 60), ("render", 30)])
        progress = reporter.stage("audio")
        progress(0.5)   # halfway through the audio stage
    """

    def __init__(
        self,
        on_log: Optional[LogCallback] = None,
        on_progress: Optional[ProgressCallback] = None,
        cancel_event: Optional[threading.Event] = None,
    ) -> None:
        self._on_log = on_log
        self._on_progress = on_progress
        self.cancel_event = cancel_event or threading.Event()
        self._weights: dict = {}
        self._offsets: dict = {}
        self._total = 1.0
        self._current = 0.0
        self._last_emit = 0.0

    # -- logging -----------------------------------------------------------------
    def log(self, message: str) -> None:
        if self._on_log:
            self._on_log(message)

    # -- cancellation --------------------------------------------------------------
    def cancel(self) -> None:
        self.cancel_event.set()

    @property
    def cancelled(self) -> bool:
        return self.cancel_event.is_set()

    def check_cancel(self) -> None:
        if self.cancel_event.is_set():
            raise Cancelled("Cancelled by user")

    # -- progress ------------------------------------------------------------------
    def plan(self, stages: Sequence[Tuple[str, float]]) -> None:
        offset = 0.0
        self._weights.clear()
        self._offsets.clear()
        for name, weight in stages:
            self._weights[name] = float(weight)
            self._offsets[name] = offset
            offset += float(weight)
        self._total = offset or 1.0
        self._current = 0.0

    def stage(self, name: str, label: Optional[str] = None) -> Callable[[float], None]:
        """Return a callback that reports progress (0..1) inside the given stage."""
        label = label or name
        weight = self._weights.get(name, 0.0)
        offset = self._offsets.get(name, self._current * self._total)

        def update(fraction: float) -> None:
            fraction = min(1.0, max(0.0, fraction))
            self._emit((offset + weight * fraction) / self._total, label)

        update(0.0)
        return update

    def done(self, label: str = "Done") -> None:
        self._emit(1.0, label, force=True)

    def _emit(self, value: float, label: str, force: bool = False) -> None:
        value = max(self._current, min(1.0, value))
        now = time.monotonic()
        # Throttle UI updates to ~20 per second.
        if not force and value < 1.0 and now - self._last_emit < 0.05:
            self._current = value
            return
        self._current = value
        self._last_emit = now
        if self._on_progress:
            self._on_progress(value, label)
