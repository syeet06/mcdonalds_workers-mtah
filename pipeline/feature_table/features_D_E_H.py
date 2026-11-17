"""Per-frame feature table for the possession model: one DataFrame row per video frame.

Each group is a function of plain arrays (no data loading), so the same code runs on IDSSE:
positions are in metres from a corner, `pitch` = (length, width). Outputs are in home's attack
frame: origin at the centre spot, home attacks +x in both halves (the pitch is rotated 180 degrees
in the half where home attacks -x). Frames outside a half (half == 0) have no direction: NaN.

    D  pitch_features   where the ball is
    E  shape_features   each team's position, movement and spread
    H  add_shifts       copies of columns from earlier and later frames
"""
import warnings

import numpy as np
import pandas as pd

from pipeline.common import HOME, AWAY
from pipeline.ball_filled import forward_speed


def _home_frame(xy, half, sign, pitch):
    """(..., 2) positions -> centred, rotated so home attacks +x. `xy` has frames on axis 0."""
    s = sign[half, HOME].astype(float)
    s[s == 0] = np.nan
    s = s.reshape((-1,) + (1,) * (xy.ndim - 1))
    return (xy - np.asarray(pitch) / 2) * s


# ------------------------------------------------------------------- D: pitch position ----

def pitch_features(ball, half, sign, pitch):
    """ball_x (+ = towards the away goal), ball_y, ball_y_abs (distance from the centre line), metres."""
    xy = _home_frame(np.asarray(ball, dtype=float), half, sign, pitch)
    return pd.DataFrame({"ball_x": xy[:, 0], "ball_y": xy[:, 1], "ball_y_abs": np.abs(xy[:, 1])})


# ------------------------------------------------------------------- E: team shape ----

def _without_keeper(xy, towards_own_goal):
    """(N, k, 2) one team's players with the deepest one (nearest its own goal) set to NaN."""
    depth = xy[:, :, 0] * towards_own_goal
    deepest = np.argmax(np.where(np.isnan(depth), -np.inf, depth), axis=1)
    out = xy.copy()
    rows = np.flatnonzero(np.isfinite(depth).any(axis=1))
    out[rows, deepest[rows]] = np.nan
    return out


def shape_features(players, team, half, ball, sign, pitch):
    """Each team's outfield centroid, forward speed, spread; players goal-side of the ball.

    Centroid and spread leave out each team's deepest player (normally the keeper); counts keep
    everyone. The goal-side counts are NaN where the ball is missing.
    """
    xy = _home_frame(np.asarray(players, dtype=float), half, sign, pitch)
    ball_x = _home_frame(np.asarray(ball, dtype=float), half, sign, pitch)[:, 0]
    fwd = forward_speed(players, team, half)
    out = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)                   # frames with nobody
        for t, name, own_goal in ((HOME, "home", -1), (AWAY, "away", 1)):
            mine = xy[:, team == t]
            field = _without_keeper(mine, own_goal)
            out[f"centroid_x_{name}"] = np.nanmean(field[:, :, 0], axis=1)
            out[f"centroid_vx_{name}"] = fwd[:, t]
            out[f"spread_{name}"] = np.sqrt(np.nanvar(field[:, :, 0], axis=1) + np.nanvar(field[:, :, 1], axis=1))
            out[f"n_tracked_{name}"] = np.isfinite(players[:, team == t, 0]).sum(axis=1)
            goal_side = (mine[:, :, 0] - ball_x[:, None]) * own_goal > 0
            out[f"n_{name}_ahead_of_ball"] = np.where(np.isnan(ball_x), np.nan, goal_side.sum(axis=1))
    df = pd.DataFrame(out)
    df["centroid_x_diff"] = df["centroid_x_home"] - df["centroid_x_away"]
    df["spread_diff"] = df["spread_home"] - df["spread_away"]
    df["n_ahead_diff"] = df["n_home_ahead_of_ball"] - df["n_away_ahead_of_ball"]
    return df


# ------------------------------------------------------------------- H: time shifts ----

def add_shifts(df, cols, half, lags=(10, 25)):
    """Add `<col>_prev<k>` / `<col>_next<k>` (value k frames before / after); NaN across half-time."""
    groups = df[cols].groupby(pd.Series(half, index=df.index))
    out = {}
    for k in lags:
        for c in cols:
            out[f"{c}_prev{k}"] = groups[c].shift(k)
            out[f"{c}_next{k}"] = groups[c].shift(-k)
    return pd.concat([df, pd.DataFrame(out, index=df.index)], axis=1)
