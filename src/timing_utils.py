"""
Timing utilities for tracking wall-clock breakdowns across training phases.
"""

from __future__ import annotations

from collections import defaultdict
from contextlib import contextmanager
import time
from typing import Dict, Iterable, Iterator, Optional

try:
    import torch
except ImportError:  # pragma: no cover - torch is optional at import time
    torch = None


DEFAULT_TIMING_KEYS = (
    "epoch_wall",
    "env_interaction",
    "imagination_rollout",
    "diffusion_sampling_teacher",
    "diffusion_sampling_student",
    "policy_value_update",
    "world_model_update",
    "distillation_oracle_query",
)


class TimingTracker:
    def __init__(self, sync_cuda: bool = False) -> None:
        self._sync_cuda = sync_cuda
        self._durations = defaultdict(float)
        self._counts = defaultdict(int)

    def reset(self) -> None:
        self._durations.clear()
        self._counts.clear()

    def add(self, name: str, seconds: float) -> None:
        self._durations[name] += seconds
        self._counts[name] += 1

    @contextmanager
    def time(self, name: str, sync_cuda: Optional[bool] = None) -> Iterator[None]:
        start = self._now(sync_cuda)
        try:
            yield
        finally:
            self.add(name, self._now(sync_cuda) - start)

    def to_log(self, keys: Optional[Iterable[str]] = None, prefix: str = "timing/") -> Dict[str, float]:
        if keys is None:
            keys = self._durations.keys()
        log: Dict[str, float] = {}
        for key in keys:
            log[f"{prefix}{key}_sec"] = float(self._durations.get(key, 0.0))
            log[f"{prefix}{key}_count"] = float(self._counts.get(key, 0))
        return log

    def _now(self, sync_cuda: Optional[bool] = None) -> float:
        if sync_cuda is None:
            sync_cuda = self._sync_cuda
        if sync_cuda and torch is not None and torch.cuda.is_available():
            torch.cuda.synchronize()
        return time.perf_counter()
