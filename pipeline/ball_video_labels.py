"""Training labels for the WASB fine-tune: the tracking ball, projected into the image and snapped to
the visible ball. Made by an algorithm, so allowed (RULES.md 1).

    from pipeline import ball_video_labels as L
    labels = L.build_labels()          # every 5th graded frame with a tracking ball -> labels.parquet
    masks = L.gap_like(trk)            # visible frames that look like gaps: contested, fast, near goal

The projected tracking ball misses the visible ball by 5-35 px, so it is snapped to the nearest small,
round, white blob. Snapping to the brightest pixel picks white boots instead.
"""

import os

import cv2
import numpy as np
import pandas as pd

from pipeline.common import CACHE_DIR, FPS, PITCH

# Params
SNAP_RADIUS = 40                    # px around the projected tracking ball searched for the ball
WHITE_LO, WHITE_HI = (0, 0, 170), (180, 70, 255)   # HSV range of the (white) ball
BALL_AREA = (12, 80)                # px; the ball is 6-7 px across
BALL_ASPECT = (0.6, 1.6)            # width / height of a round blob
BALL_FILL = 0.55                    # blob area / bounding-box area
CONTESTED_M = 2.0                   # players this close to the ball count as contesting it
CONTESTED_N = 2
FAST_MS = 15.0
SPEED_SPAN = 2                      # frames either side for the ball speed (centred difference)
BOX_DEPTH_M, BOX_HALF_WIDTH_M = 16.5, 20.16
LABEL_STEP = 5                      # label every 5th eligible frame
MAX_READ_THROUGH = 50               # read through gaps up to this many frames instead of seeking
LABELS_PATH = os.path.join(CACHE_DIR, "ball_video", "labels.parquet")


def snap_ball(img, u, v, radius=SNAP_RADIUS):
    """(x, y) of the small, round, white blob nearest (u, v), or None if there is none or the search
    window does not fit inside the image."""
    x0, y0 = int(round(u)) - radius, int(round(v)) - radius
    if x0 < 0 or y0 < 0 or x0 + 2 * radius > img.shape[1] or y0 + 2 * radius > img.shape[0]:
        return None
    hsv = cv2.cvtColor(img[y0:y0 + 2 * radius, x0:x0 + 2 * radius], cv2.COLOR_BGR2HSV)
    n, _, stats, cent = cv2.connectedComponentsWithStats(cv2.inRange(hsv, WHITE_LO, WHITE_HI))
    best, best_d = None, np.inf
    for k in range(1, n):
        _, _, w, h, area = stats[k]
        round_enough = BALL_ASPECT[0] < w / h < BALL_ASPECT[1] and area > BALL_FILL * w * h
        if BALL_AREA[0] <= area <= BALL_AREA[1] and round_enough:
            d = np.hypot(cent[k][0] - radius, cent[k][1] - radius)
            if d < best_d:
                best, best_d = (x0 + int(round(cent[k][0])), y0 + int(round(cent[k][1]))), d
    return best


def px_per_metre(H, xy_n):
    """Image pixels per metre of pitch, horizontally, at a normalised pitch position."""
    from starter.load_homography import project
    a = np.array(project(H, *xy_n))
    b = np.array(project(H, xy_n[0] + 1 / PITCH[0], xy_n[1]))
    return float(np.hypot(*(b - a)))


def ball_speed_ms(ball, span=SPEED_SPAN):
    """(N,) ball speed in m/s from positions, teleports removed; NaN near missing frames. The ATD's own
    ball speed column is missing on 91% of visible frames and does not match the positions."""
    from pipeline.ball_filled import remove_teleports

    ball, _ = remove_teleports(ball)
    speed = np.full(len(ball), np.nan)
    speed[span:-span] = np.linalg.norm(ball[2 * span:] - ball[:-2 * span], axis=1) / (2 * span) * FPS
    return speed


def gap_like(trk):
    """Masks of frames whose ball situation resembles a gap: contested, fast, near a goal."""
    ball = trk["ball"]
    with np.errstate(invalid="ignore"):
        dist = np.linalg.norm(trk["players"] - ball[:, None], axis=2)
        contested = (dist < CONTESTED_M).sum(axis=1) >= CONTESTED_N
        fast = ball_speed_ms(ball) > FAST_MS
        in_x = (ball[:, 0] < BOX_DEPTH_M) | (ball[:, 0] > PITCH[0] - BOX_DEPTH_M)
        near_goal = in_x & (np.abs(ball[:, 1] - PITCH[1] / 2) < BOX_HALF_WIDTH_M)
    return {"contested": contested, "fast": fast, "near_goal": near_goal}


def label_table(trk, H, reader, frames):
    """One row per frame: projected tracking ball (u, v), snapped ball (x, y; NaN if not snapped),
    snapped, fold (10-minute chunk) and pixels per metre at the ball."""
    from pipeline.ball_impute import chunk_fold
    from pipeline.common import half_of_frame
    from starter.load_homography import project

    frames = np.sort(np.asarray(frames))
    ball_n = trk["ball"] / np.array(PITCH)
    rows, last = [], None
    for f in frames:
        f = int(f)
        if last is not None and 0 < f - last <= MAX_READ_THROUGH:
            img = reader.read(last + 1, f - last)[-1]          # decoding through beats seeking
        else:
            img = reader.read(f, 1)[0]
        last = f
        u, v = project(H[f], *ball_n[f])
        snapped = snap_ball(img, u, v)
        x, y = snapped if snapped is not None else (np.nan, np.nan)
        rows.append((f, float(u), float(v), x, y, snapped is not None, px_per_metre(H[f], ball_n[f])))
    table = pd.DataFrame(rows, columns=["frame", "u", "v", "x", "y", "snapped", "ppm"])
    table.insert(6, "fold", chunk_fold(table["frame"].to_numpy(), half_of_frame()))
    return table


def build_labels(step=LABEL_STEP):
    """Label every `step`-th graded frame where the tracking has the ball; saved to LABELS_PATH."""
    from games import require
    from pipeline.ball_video_io import FrameReader
    from pipeline.common import graded_mask, load_tracking
    from starter.load_homography import load

    trk = load_tracking()
    H, _, _ = load()
    frames = np.flatnonzero(graded_mask() & ~np.isnan(trk["ball"][:, 0]))[::step]
    table = label_table(trk, H, FrameReader(require(["video"])[0]), frames)
    os.makedirs(os.path.dirname(LABELS_PATH), exist_ok=True)
    table.to_parquet(LABELS_PATH)
    return table
