"""When the TV broadcast is not showing play: replays, close-ups and crowd shots, on tactical frames.

    from pipeline.broadcast import away_starts
    starts = away_starts(half)       # tactical frames where the broadcast cuts away from play

The broadcast video has its own timeline (DATA.md: not synchronised, no mapping given). Live TV runs
in real time, so a replay uses up real time while the match clock keeps running, and the broadcast ->
tactical offset is constant within each half. The offsets below were found by matching the ball's
pitch position in the two tracking files: median distance under 1 m.
check_alignment() re-measures them and refuses to go on if they no longer fit.

The broadcast homography is invalid exactly where the picture is not the pitch (34% of its frames).
Mapped onto tactical frames, the TV is away from play on 31% of graded frames; on the public five
minutes 85% of those are dead. Close-ups during live play make up the rest, so the submission uses
each cut as an anchor (dead_refine.anchor_set_pieces: needs the ball to go quiet and restart), not
as dead frames. Public window: dead-ball F1 0.893 -> 0.932, recall 0.906 -> 0.994.
"""

import os

import numpy as np
import pandas as pd

from games import asset_paths
from pipeline.common import CACHE_DIR, N_FRAMES, PITCH, load_tracking, runs

# Params
SEGMENTS = ((0, 84700, 4014), (86800, None, 26920))   # (first, end) broadcast frames, tactical = broadcast + offset
MAX_MEDIAN_M = 1.5                                       # alignment check: ball distance between the two trackings
CACHE = os.path.join(CACHE_DIR, "broadcast_tracking.npz")


def load_broadcast():
    """Broadcast ball (M, 2) metres and homography valid (M,) bool. Parsed once, then cached."""
    if os.path.exists(CACHE):
        with np.load(CACHE) as f:
            return f["ball"], f["valid"]
    paths = asset_paths(source="broadcast")
    with open(paths["atd_raw"]) as handle:
        lines = handle.read().splitlines()
    ball = np.array([[float(v) for v in line.rsplit(":", 1)[-1].split(",")[:2]] for line in lines]) * np.array(PITCH)
    valid = pd.read_csv(paths["homography"], usecols=["valid"])["valid"].to_numpy().astype(bool)
    if len(valid) != len(ball):
        raise ValueError("broadcast tracking and homography have different frame counts")
    np.savez(CACHE, ball=ball, valid=valid)
    return ball, valid


def tactical_map(n_broadcast, segments=SEGMENTS, n_frames=N_FRAMES):
    """(broadcast frames, tactical frames) for every broadcast frame inside a segment and the video."""
    b, t = [], []
    for first, end, offset in segments:
        frames = np.arange(first, n_broadcast if end is None else end)
        keep = (frames + offset >= 0) & (frames + offset < n_frames)
        b.append(frames[keep])
        t.append(frames[keep] + offset)
    return np.concatenate(b), np.concatenate(t)


def check_alignment(b_ball, t_ball, segments=SEGMENTS):
    """Median ball distance (m) per segment; ValueError if any is over MAX_MEDIAN_M."""
    out = []
    for segment in segments:
        b, t = tactical_map(len(b_ball), (segment,), len(t_ball))
        dist = np.linalg.norm(b_ball[b] - t_ball[t], axis=1)
        out.append(float(np.median(dist[np.isfinite(dist)])))
    if max(out) > MAX_MEDIAN_M:
        raise ValueError(f"broadcast alignment no longer fits: median ball distance {out} m")
    return out


def away_mask(valid, half, segments=SEGMENTS):
    """(N,) bool: graded tactical frames where the broadcast shows no pitch (homography invalid)."""
    b, t = tactical_map(len(valid), segments, len(half))
    away = np.zeros(len(half), dtype=bool)
    away[t] = ~valid[b]
    return away & (half > 0)


def away_starts(half):
    """Tactical frames where the broadcast cuts away from play, after checking the alignment."""
    ball, valid = load_broadcast()
    check_alignment(ball, load_tracking()["ball"].astype(float))
    return np.array([s for value, s, _ in runs(away_mask(valid, half)) if value], dtype=int)
