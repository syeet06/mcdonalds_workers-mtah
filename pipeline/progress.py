"""Live progress on the terminal: numbered steps with timings, and a bar for the long loops.

    from pipeline import progress
    with progress.step("Dead-ball mask"):                # [5/7] Dead-ball mask ... done in 12.3 s
        ...
    bar = progress.Bar(n, "reading the video")           #   reading the video |█████░░░░░|  52%  ...
    for i in ...: bar.update(i + 1)
    bar.close()

Only prints; nothing here changes what the pipeline computes. No dependencies, so the scorer's
environment can run it. Off a terminal (a log file, CI) the bar prints a line every 10% instead of
redrawing itself.
"""

import sys
import time
from contextlib import contextmanager

TOTAL_STEPS = 7                 # pipeline.submission's numbered steps
WIDTH = 28                      # bar width in characters
REDRAW_S = 0.2                  # redraw the bar at most this often

_count = [0]
_start = [None]


def _out(text, end="\n"):
    sys.stdout.write(text + end)
    sys.stdout.flush()


def _clock(seconds):
    seconds = int(round(seconds))
    return f"{seconds // 3600}:{seconds % 3600 // 60:02d}:{seconds % 60:02d}" if seconds >= 3600 \
        else f"{seconds // 60}:{seconds % 60:02d}"


def begin(title):
    """The run's header; resets the step count."""
    _count[0], _start[0] = 0, time.monotonic()
    _out(f"\n{title}\n" + "─" * len(title))


@contextmanager
def step(text):
    """One numbered step: prints what it is doing, then how long it took."""
    _count[0] += 1
    _out(f"\n[{_count[0]}/{TOTAL_STEPS}] {text}")
    t0 = time.monotonic()
    yield
    _out(f"      done in {_clock(time.monotonic() - t0)}")


def note(text):
    """A detail line inside the current step."""
    _out(f"      {text}")


def finish():
    if _start[0] is not None:
        _out(f"\nAll done in {_clock(time.monotonic() - _start[0])}.")


class Bar:
    """A progress bar for a loop of `total` items, with elapsed time and an estimate of what's left."""

    def __init__(self, total, label):
        self.total, self.label = max(int(total), 1), label
        self.t0 = self.last = time.monotonic()
        self.tty = sys.stdout.isatty()
        self.decile = -1
        self.update(0)

    def update(self, done):
        now = time.monotonic()
        frac = min(max(done / self.total, 0.0), 1.0)
        if self.tty:
            if now - self.last < REDRAW_S and frac < 1.0 and done:
                return
            self.last = now
            _out("\r" + self._line(frac, now), end="")
        elif int(frac * 10) > self.decile:            # a log file: one line per 10%
            self.decile = int(frac * 10)
            _out(self._line(frac, now))

    def _line(self, frac, now):
        filled = int(round(frac * WIDTH))
        elapsed = now - self.t0
        left = f" · ~{_clock(elapsed / frac - elapsed)} left" if 0 < frac < 1 and elapsed > 2 else ""
        return (f"      {self.label} |{'█' * filled}{'░' * (WIDTH - filled)}| {frac:4.0%}"
                f" · {_clock(elapsed)} elapsed{left}")

    def close(self):
        self.update(self.total)
        if self.tty:
            _out("")
