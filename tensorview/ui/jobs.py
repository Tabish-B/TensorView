"""Serialize file operations and deliver results on the Qt UI thread."""

from threading import Event
from time import monotonic

from PySide6.QtCore import QObject, QRunnable, QThreadPool, Signal, Slot


class JobCancelled(Exception):
    pass


class _Worker(QRunnable):
    def __init__(self, bridge, token, task, cancelled):
        super().__init__()
        self.bridge, self.token, self.task, self.cancelled = bridge, token, task, cancelled
        self.last_update = 0.0

    def progress(self, completed, total):
        if self.cancelled.is_set():
            raise JobCancelled
        current = monotonic()
        if current - self.last_update >= 0.08 or completed == total:
            self.last_update = current
            self.bridge.advanced.emit(self.token, completed, total)

    def run(self):
        value = error = None
        if not self.cancelled.is_set():
            try:
                value = self.task(self.progress)
            except Exception as exception:
                error = exception
        self.bridge.finished.emit(self.token, value, error)


class Jobs(QObject):
    finished = Signal(int, object, object)
    advanced = Signal(int, object, object)
    busy = Signal(str, object, object)
    error = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.pool = QThreadPool(self)
        self.pool.setMaxThreadCount(1)
        self.token = 0
        self.current = {}
        self.pending = {}
        self.foreground = {}
        self.finished.connect(self._finish)
        self.advanced.connect(self._progress)

    def submit(self, group, task, callback, *, message="", error_callback=None, replace=True, priority=0):
        if replace:
            self.cancel(group)
        self.token += 1
        number = self.token
        cancelled = Event()
        self.current[group] = (number, cancelled)
        self.pending[number] = (group, cancelled, callback, error_callback)
        if message:
            self.foreground[number] = [message, 0, 0]
            self._refresh()
        self.pool.start(_Worker(self, number, task, cancelled), priority)
        return number

    def cancel(self, group):
        previous = self.current.pop(group, None)
        if previous:
            token, flag = previous
            flag.set()
            if self.foreground.pop(token, None):
                self._refresh()

    def cancel_all(self):
        for group in list(self.current):
            self.cancel(group)
        for _, flag, _, _ in self.pending.values():
            flag.set()
        self.pool.clear()

    def _refresh(self):
        state = next(reversed(self.foreground.values()), ["", 0, 0])
        self.busy.emit(*state)

    @Slot(int, object, object)
    def _finish(self, token, value, error):
        group, flag, callback, on_error = self.pending.pop(token)
        if self.current.get(group, (None,))[0] == token:
            self.current.pop(group)
        if self.foreground.pop(token, None):
            self._refresh()
        if flag.is_set() or isinstance(error, JobCancelled):
            return
        if error is None:
            callback(value)
        elif on_error:
            on_error(str(error))
        else:
            self.error.emit(str(error))

    @Slot(int, object, object)
    def _progress(self, token, completed, total):
        if token in self.foreground:
            self.foreground[token][1:] = completed, total
            self._refresh()
