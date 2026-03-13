"""Callback system for training progress monitoring and control."""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Callable
import time
import json


class Callback(ABC):
    """Base class for training callbacks."""

    def on_train_begin(self, logs: Optional[Dict[str, Any]] = None) -> None:
        """Called at the start of training."""
        pass

    def on_train_end(self, logs: Optional[Dict[str, Any]] = None) -> None:
        """Called at the end of training."""
        pass

    def on_epoch_begin(self, epoch: int, logs: Optional[Dict[str, Any]] = None) -> None:
        """Called at the start of each epoch."""
        pass

    def on_epoch_end(self, epoch: int, logs: Optional[Dict[str, Any]] = None) -> bool:
        """
        Called at the end of each epoch.

        Returns
        -------
        bool
            True to continue training, False to stop early
        """
        return True

    def on_phase_begin(self, phase: str, logs: Optional[Dict[str, Any]] = None) -> None:
        """Called at the start of a training phase (pretrain/ssc)."""
        pass

    def on_phase_end(self, phase: str, logs: Optional[Dict[str, Any]] = None) -> None:
        """Called at the end of a training phase."""
        pass


class CallbackList:
    """Container for managing multiple callbacks."""

    def __init__(self, callbacks: Optional[List[Callback]] = None):
        self.callbacks = callbacks or []

    def append(self, callback: Callback) -> None:
        self.callbacks.append(callback)

    def on_train_begin(self, logs: Optional[Dict[str, Any]] = None) -> None:
        for cb in self.callbacks:
            cb.on_train_begin(logs)

    def on_train_end(self, logs: Optional[Dict[str, Any]] = None) -> None:
        for cb in self.callbacks:
            cb.on_train_end(logs)

    def on_epoch_begin(self, epoch: int, logs: Optional[Dict[str, Any]] = None) -> None:
        for cb in self.callbacks:
            cb.on_epoch_begin(epoch, logs)

    def on_epoch_end(self, epoch: int, logs: Optional[Dict[str, Any]] = None) -> bool:
        continue_training = True
        for cb in self.callbacks:
            if not cb.on_epoch_end(epoch, logs):
                continue_training = False
        return continue_training

    def on_phase_begin(self, phase: str, logs: Optional[Dict[str, Any]] = None) -> None:
        for cb in self.callbacks:
            cb.on_phase_begin(phase, logs)

    def on_phase_end(self, phase: str, logs: Optional[Dict[str, Any]] = None) -> None:
        for cb in self.callbacks:
            cb.on_phase_end(phase, logs)


class EarlyStopping(Callback):
    """
    Stop training when a monitored metric stops improving.

    Parameters
    ----------
    monitor : str
        Metric name to monitor (e.g., 'loss', 'delta')
    patience : int
        Number of epochs with no improvement before stopping
    min_delta : float
        Minimum change to qualify as an improvement
    mode : str
        'min' for metrics to minimize, 'max' for metrics to maximize
    """

    def __init__(
        self,
        monitor: str = 'loss',
        patience: int = 10,
        min_delta: float = 0.0,
        mode: str = 'min',
        verbose: bool = True
    ):
        self.monitor = monitor
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.verbose = verbose

        self.best_value: Optional[float] = None
        self.wait = 0
        self.stopped_epoch = 0

        if mode == 'min':
            self.monitor_op = lambda a, b: a < b - min_delta
        else:
            self.monitor_op = lambda a, b: a > b + min_delta

    def on_train_begin(self, logs: Optional[Dict[str, Any]] = None) -> None:
        self.best_value = None
        self.wait = 0
        self.stopped_epoch = 0

    def on_epoch_end(self, epoch: int, logs: Optional[Dict[str, Any]] = None) -> bool:
        logs = logs or {}
        current = logs.get(self.monitor)

        if current is None:
            return True

        if self.best_value is None or self.monitor_op(current, self.best_value):
            self.best_value = current
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.stopped_epoch = epoch
                if self.verbose:
                    print(f"Early stopping at epoch {epoch + 1}")
                return False

        return True


class ProgressLogger(Callback):
    """
    Log training progress to console or file.

    Parameters
    ----------
    log_interval : int
        Log every N epochs
    log_file : str, optional
        Path to log file (JSON format)
    metrics : list, optional
        Metrics to log (logs all if None)
    """

    def __init__(
        self,
        log_interval: int = 10,
        log_file: Optional[str] = None,
        metrics: Optional[List[str]] = None
    ):
        self.log_interval = log_interval
        self.log_file = log_file
        self.metrics = metrics
        self.history: List[Dict[str, Any]] = []
        self.current_phase: str = ""

    def on_phase_begin(self, phase: str, logs: Optional[Dict[str, Any]] = None) -> None:
        self.current_phase = phase

    def on_epoch_end(self, epoch: int, logs: Optional[Dict[str, Any]] = None) -> bool:
        logs = logs or {}

        record = {'epoch': epoch + 1, 'phase': self.current_phase}
        if self.metrics:
            record.update({k: logs.get(k) for k in self.metrics if k in logs})
        else:
            record.update(logs)

        self.history.append(record)

        if (epoch + 1) % self.log_interval == 0:
            metrics_str = ", ".join(f"{k}={v:.4f}" for k, v in logs.items() if isinstance(v, (int, float)))
            print(f"[{self.current_phase}] Epoch {epoch + 1}: {metrics_str}")

        return True

    def on_train_end(self, logs: Optional[Dict[str, Any]] = None) -> None:
        if self.log_file:
            with open(self.log_file, 'w') as f:
                json.dump(self.history, f, indent=2)

    def get_history(self) -> List[Dict[str, Any]]:
        return self.history


class Timer(Callback):
    """Track and report training time."""

    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.train_start: Optional[float] = None
        self.phase_start: Optional[float] = None
        self.epoch_times: List[float] = []
        self.phase_times: Dict[str, float] = {}

    def on_train_begin(self, logs: Optional[Dict[str, Any]] = None) -> None:
        self.train_start = time.time()
        self.epoch_times = []
        self.phase_times = {}

    def on_phase_begin(self, phase: str, logs: Optional[Dict[str, Any]] = None) -> None:
        self.phase_start = time.time()

    def on_phase_end(self, phase: str, logs: Optional[Dict[str, Any]] = None) -> None:
        if self.phase_start:
            elapsed = time.time() - self.phase_start
            self.phase_times[phase] = elapsed
            if self.verbose:
                print(f"{phase} completed in {elapsed:.2f}s")

    def on_epoch_begin(self, epoch: int, logs: Optional[Dict[str, Any]] = None) -> None:
        self._epoch_start = time.time()

    def on_epoch_end(self, epoch: int, logs: Optional[Dict[str, Any]] = None) -> bool:
        if hasattr(self, '_epoch_start'):
            self.epoch_times.append(time.time() - self._epoch_start)
        return True

    def on_train_end(self, logs: Optional[Dict[str, Any]] = None) -> None:
        if self.verbose and self.train_start:
            total = time.time() - self.train_start
            print(f"Total training time: {total:.2f}s")
            if self.epoch_times:
                avg = sum(self.epoch_times) / len(self.epoch_times)
                print(f"Average epoch time: {avg:.4f}s")

    def get_stats(self) -> Dict[str, Any]:
        return {
            'total_time': time.time() - self.train_start if self.train_start else 0,
            'phase_times': self.phase_times,
            'avg_epoch_time': sum(self.epoch_times) / len(self.epoch_times) if self.epoch_times else 0
        }


class LambdaCallback(Callback):
    """
    Create callbacks from lambda functions.

    Parameters
    ----------
    on_epoch_end : callable, optional
        Function called at end of each epoch: f(epoch, logs) -> bool
    on_train_end : callable, optional
        Function called at end of training: f(logs)
    """

    def __init__(
        self,
        on_epoch_end_fn: Optional[Callable[[int, Dict[str, Any]], bool]] = None,
        on_train_end_fn: Optional[Callable[[Dict[str, Any]], None]] = None
    ):
        self._on_epoch_end = on_epoch_end_fn
        self._on_train_end = on_train_end_fn

    def on_epoch_end(self, epoch: int, logs: Optional[Dict[str, Any]] = None) -> bool:
        if self._on_epoch_end:
            return self._on_epoch_end(epoch, logs or {})
        return True

    def on_train_end(self, logs: Optional[Dict[str, Any]] = None) -> None:
        if self._on_train_end:
            self._on_train_end(logs or {})


class CheckpointCallback(Callback):
    """
    Save model checkpoints during training.

    Parameters
    ----------
    save_fn : callable
        Function to save the model: f(path)
    filepath : str
        Path template for checkpoints (use {epoch} for epoch number)
    save_freq : int
        Save every N epochs
    save_best_only : bool
        Only save when monitored metric improves
    monitor : str
        Metric to monitor for best model
    mode : str
        'min' or 'max' for the monitored metric
    """

    def __init__(
        self,
        save_fn: Callable[[str], None],
        filepath: str = "checkpoint_{epoch}.pt",
        save_freq: int = 10,
        save_best_only: bool = False,
        monitor: str = 'loss',
        mode: str = 'min'
    ):
        self.save_fn = save_fn
        self.filepath = filepath
        self.save_freq = save_freq
        self.save_best_only = save_best_only
        self.monitor = monitor
        self.mode = mode

        self.best_value: Optional[float] = None
        if mode == 'min':
            self.is_better = lambda a, b: a < b
        else:
            self.is_better = lambda a, b: a > b

    def on_epoch_end(self, epoch: int, logs: Optional[Dict[str, Any]] = None) -> bool:
        logs = logs or {}

        if self.save_best_only:
            current = logs.get(self.monitor)
            if current is not None:
                if self.best_value is None or self.is_better(current, self.best_value):
                    self.best_value = current
                    path = self.filepath.format(epoch=epoch + 1)
                    self.save_fn(path)
        elif (epoch + 1) % self.save_freq == 0:
            path = self.filepath.format(epoch=epoch + 1)
            self.save_fn(path)

        return True
