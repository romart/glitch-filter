"""Throttled terminal progress, kept separate from machine-readable stdout."""

import sys
import time


def duration(seconds):
    seconds = max(0, int(seconds))
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours}:{minutes:02}:{seconds:02}" if hours else f"{minutes}:{seconds:02}"


class Progress:
    def __init__(self, label, total=None, *, enabled=True, stream=None, clock=None):
        self.label, self.total, self.enabled = label, total, enabled
        self.stream = stream if stream is not None else sys.stderr
        self.clock = clock if clock is not None else time.monotonic
        self.completed = 0
        self.started = self.clock()
        self.last_render = self.started

    def __enter__(self):
        self.render(force=True)
        return self

    def advance(self):
        self.completed += 1
        self.render()

    def render(self, *, force=False, status=None):
        if not self.enabled:
            return
        now = self.clock()
        if not force and now - self.last_render < 1.0:
            return
        elapsed = max(0, now - self.started)
        rate = self.completed / elapsed if elapsed else 0
        count = str(self.completed)
        eta = "--"
        if self.total is not None:
            fraction = self.completed / self.total if self.total else 1.0
            filled = min(20, int(fraction * 20))
            count = f"[{'#' * filled}{'-' * (20 - filled)}] {count}/{self.total} ({fraction:.1%})"
            if rate:
                eta = duration((self.total - self.completed) / rate)
        line = f"{self.label}: {count} | {rate:.1f} img/s | elapsed {duration(elapsed)} | ETA {eta}"
        if status:
            line += f" | {status}"
        terminal = self.stream.isatty()
        self.stream.write(("\r\033[K" if terminal else "") + line + ("\n" if status or not terminal else ""))
        self.stream.flush()
        self.last_render = now

    def __exit__(self, exc_type, exc_value, traceback):
        self.render(force=True, status="done" if exc_type is None else "stopped")
