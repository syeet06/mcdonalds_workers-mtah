"""Per-frame clues for the decoder: (N,) scores, positive = HOME has the ball, negative = AWAY, 0 = no opinion.

    from pipeline.clues import ball_clue, tag_intervals, tag_weights, tag_clue
"""

import os
import sys
import warnings

import numpy as np
import pandas as pd

from pipeline.ball_filled import clean_team
from pipeline.common import AWAY, HOME, N_FRAMES, NOBODY, ROOT

# Params
W_CLEAN = 3.0           # ball clearly at one team's feet
W_NEAR = 0.5            # ball visible, but contested
NEAR_SCALE_M = 2.0      # distance difference that counts as "clearly closer"
TAG_CAP = 2.5           # largest weight one tag can carry
TAG_MIN_AGREEMENT = 0.55  # codes no better than this get weight 0

TEAMS = {"BARCELONA": HOME, "ATLETIC CLUB": AWAY}
ATTACKING = {"BUILD UP", "PROGRESSION", "CREATING CHANCES", "BALL IN FINAL THIRD", "BALL IN THE BOX",
             "PLAYERS IN THE BOX", "LONG BALL", "ATTACKING TRANSITION"}
DEFENDING = {"DEFENDING IN ATTACKING THIRD", "DEFENDING IN MIDDLE THIRD",
             "DEFENDING IN DEFENSIVE THIRD", "DEFENSIVE TRANSITION"}


def _closeness(ball, players, team):
    """(N,) tanh of how much closer the nearest HOME player is than the nearest AWAY one; NaN if unknown."""
    distance = np.linalg.norm(players - ball[:, None, :], axis=2)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)                   # frames with nobody
        home = np.nanmin(np.where(team == HOME, distance, np.nan), axis=1)
        away = np.nanmin(np.where(team == AWAY, distance, np.nan), axis=1)
    return np.tanh((away - home) / NEAR_SCALE_M)


def ball_clue(ball, players, team, w_clean=W_CLEAN, w_near=W_NEAR, **clean_kw):
    """Strong vote on clean frames, a small distance-scaled vote when visible but contested, 0 when missing.
    `clean_kw` (control, margin) goes to clean_team."""
    score = np.nan_to_num(w_near * _closeness(ball, players, team), nan=0.0)
    clean = clean_team(ball, players, team, **clean_kw)
    score[clean == HOME] = w_clean
    score[clean == AWAY] = -w_clean
    return score


def inferred_ball_clue(ball_inferred, players, team, w_inf):
    """Vote from the inferred ball inside gaps (pipeline.ball_impute): which team is closer to it.
    0 where nothing was inferred."""
    return np.nan_to_num(w_inf * _closeness(ball_inferred, players, team), nan=0.0)


def tag_intervals(events=None, n_frames=N_FRAMES):
    """Smart Tagging phases that say which team has the ball: DataFrame[code, start, end, has].

    Attacking codes credit the tagged team, defending codes the other one. Tags without a team
    (N/A: throw-ins, kick-offs) and set pieces are dropped. A phase tagged from both sides (e.g. Barça
    PROGRESSION and Athletic DEFENDING IN MIDDLE THIRD over the same frames) is kept once.
    """
    if events is None:
        sys.path.insert(0, os.path.join(ROOT, "starter"))
        from load_smart_tagging import load
        events = load()
    rows = []
    for event in events:
        code = event["code"].upper()
        team = TEAMS.get(str(event["tags"].get("Team", "")).upper())
        if team is None or code not in ATTACKING | DEFENDING:
            continue
        has = team if code in ATTACKING else 1 - team
        rows.append({"code": code, "start": int(event["start_frame"]), "end": int(event["end_frame"]),
                     "has": has})
    out = pd.DataFrame(rows, columns=["code", "start", "end", "has"])
    return out.drop_duplicates(["start", "end", "has"]).reset_index(drop=True)


def tag_weights(intervals, clean):
    """Weight per code = log(p / (1 - p)), p = how often the team it implies has the ball at its feet
    (clean frames, whole match, no ground truth). Capped at TAG_CAP; 0 if p <= TAG_MIN_AGREEMENT."""
    agree, total = {}, {}
    for row in intervals.itertuples():
        frames = clean[max(row.start, 0):max(row.end, 0)]
        frames = frames[frames != NOBODY]
        agree[row.code] = agree.get(row.code, 0) + int(np.sum(frames == row.has))
        total[row.code] = total.get(row.code, 0) + len(frames)
    weights = {}
    for code in total:
        p = agree[code] / total[code] if total[code] else 0.5
        if p <= TAG_MIN_AGREEMENT:
            weights[code] = 0.0
        else:
            weights[code] = float(min(np.log(p / (1 - p)) if p < 1 else TAG_CAP, TAG_CAP))
    return weights


def tag_clue(intervals, weights, n_frames=N_FRAMES):
    """Sum of tag votes per frame: + for HOME, - for AWAY. Intervals are clipped to the array."""
    score = np.zeros(n_frames)
    for row in intervals.itertuples():
        start, end = max(row.start, 0), min(row.end, n_frames)
        if end > start:
            score[start:end] += weights.get(row.code, 0.0) * (1 if row.has == HOME else -1)
    return score
