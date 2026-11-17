"""Build the whole per-frame feature table (groups A-F, plus H time shifts) for any match.

    from pipeline.feature_table.build_features import build_feature_table, load_our_match

    table = build_feature_table(load_our_match())          # our match, live frames only
    save_feature_table(table)                              # pipeline/cache/feature_table.pkl + .csv

Or from the repo root: python -m pipeline.feature_table.build_features

The feature code never loads data: everything goes through a Match, a bundle of plain arrays.
To run IDSSE (or any other source) through exactly the same code, fill a Match yourself:

    match = Match(
        ball=ball_xy,                # (N, 2) metres from a corner; NaN where missing
        players=player_xy,           # (N, S, 2) metres from a corner; NaN where a slot is empty
        team=slot_team,              # (S,) HOME (0) or AWAY (1) for each slot
        half=half_id,                # (N,) 1, 2, or 0 outside play
        pitch=(105.0, 68.0),         # (length, width) in metres
        live=ball_alive,             # (N,) bool, optional: rows outside it are dropped
        label=home_has_ball,         # (N,) 1 home / 0 away / NaN, optional: added as `label`
    )
    table = build_feature_table(match)

IDSSE positions are centred on the centre spot; convert them with from_centre_origin() first.
ball_source is optional: without it, every frame with a ball position counts as observed.

Groups and where they come from
-------------------------------
A  who is near the ball        features_A_B.distance_features
B  control zones and memory    features_A_B.control_features
C  how the ball moves          features_C_D_F.ball_motion_features
D  where on the pitch          features_D_E_H.pitch_features   (home attacks +x in both halves)
E  team shape                  features_D_E_H.shape_features
F  how much to trust the ball  features_C_D_F.trust_features
H  time-shifted copies         features_D_E_H.add_shifts       (never across half-time)

D comes from features_D_E_H, not features_C_D_F.position_features: it turns the pitch so home
always attacks +x, which keeps ball_x meaning the same thing in both halves and in every match.
Each output row also carries `frame` (row in the original arrays) and `half`.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from pipeline.ball_filled import MISSING, OBSERVED, attack_sign
from pipeline.common import AWAY, FPS, HOME, PITCH
from pipeline.feature_table.features_A_B import control_features, distance_features
from pipeline.feature_table.features_C_D_F import ball_motion_features, trust_features
from pipeline.feature_table.features_D_E_H import add_shifts, pitch_features, shape_features

# Params
SHIFT_COLS = [                    # columns that get _prev<k> / _next<k> copies (group H)
    "margin", "is_duel", "last_control_team",   # A, B
    "ball_speed", "ball_accel",                 # C
    "ball_x",                                   # D
    "centroid_x_diff",                          # E
]
SHIFT_LAGS = (10, 25)             # frames: 0.4 s and 1 s at 25 fps


@dataclass
class Match:
    """One match as plain arrays. Positions in metres from a corner, one row per frame."""

    ball: np.ndarray                          # (N, 2)
    players: np.ndarray                       # (N, S, 2)
    team: np.ndarray                          # (S,) HOME / AWAY
    half: np.ndarray                          # (N,) 1, 2, or 0 outside play
    pitch: tuple = PITCH                      # (length, width) in metres
    fps: int = FPS
    ball_source: Optional[np.ndarray] = None  # (N,) ball_filled codes; None = observed where present
    live: Optional[np.ndarray] = None         # (N,) bool; None = keep every frame inside a half
    label: Optional[np.ndarray] = None        # (N,) 1 home / 0 away; None = no label column


def build_feature_table(match, shift_cols=SHIFT_COLS, lags=SHIFT_LAGS, drop_dead=True):
    """Return one DataFrame row per frame with every feature group, for any Match.

    Rows outside both halves are always dropped; with drop_dead, so are rows where match.live
    is False. Shifts are taken before dropping, so _prev / _next look at real neighbouring frames.
    """
    ball, players, team, half, ball_source = _checked(match)

    # Which way each team attacks in each half: [half, team] = +1 towards +x. D and E use it to
    # turn the pitch so home attacks +x; C uses the same rule to find each team's half.
    sign = attack_sign(players, team, half)

    # A and B: distances use the full (filled) ball so they agree with D and E about where the
    # ball is. C and F use ball_source themselves to keep only observed positions.
    a = distance_features(ball, players, team)
    b = control_features(a["d_home"], a["d_away"], half)
    c = ball_motion_features(ball, ball_source, players, team, half, match.pitch)
    d = pitch_features(ball, half, sign, match.pitch)
    e = shape_features(players, team, half, ball, sign, match.pitch)
    f = trust_features(ball_source, players, half)

    table = pd.concat([pd.DataFrame(a), pd.DataFrame(b), pd.DataFrame(c), d, e, pd.DataFrame(f)], axis=1)
    duplicated = table.columns[table.columns.duplicated()].tolist()
    if duplicated:
        raise ValueError(f"Two feature groups produce the same column: {duplicated}")

    # H: copies from earlier and later frames, grouped by half so nothing crosses half-time.
    table = add_shifts(table, shift_cols, half, lags)

    table.insert(0, "frame", np.arange(len(table)))
    table.insert(1, "half", half)
    if match.label is not None:
        table["label"] = np.asarray(match.label, dtype=float)

    keep = half > 0
    if drop_dead and match.live is not None:
        keep &= np.asarray(match.live, dtype=bool)
    return table[keep].reset_index(drop=True)


def load_our_match():
    """Match for the challenge game: tracking (goalkeeper tracks by the goal they stand in,
    pipeline/keepers.py), Task 3's filled ball, and Task 4's dead mask."""
    from pipeline.common import half_of_frame, load_stage, load_tracking
    from pipeline.keepers import keepers_by_goal

    trk = load_tracking()
    ball = load_stage("ball_filled")
    half = half_of_frame()
    players, team = keepers_by_goal(trk["players"], trk["team"], half)
    return Match(
        ball=ball["ball"],
        players=players,
        team=team,
        half=half,
        ball_source=ball["ball_source"],
        live=~load_stage("dead_ball")["dead"],
    )


def save_feature_table(table, name="feature_table", csv=True):
    """Write table to pipeline/cache/<name>.pkl (and .csv); return the paths written.

    The .pkl reloads exactly with pd.read_pickle (same dtypes, fast). The .csv is for looking at
    the table in VS Code, Excel or a spreadsheet. cache/ is gitignored: these files hold match
    data and must never reach the public repo.
    """
    import os

    from pipeline.common import CACHE_DIR

    os.makedirs(CACHE_DIR, exist_ok=True)
    paths = [os.path.join(CACHE_DIR, f"{name}.pkl")]
    table.to_pickle(paths[0])
    if csv:
        paths.append(os.path.join(CACHE_DIR, f"{name}.csv"))
        table.to_csv(paths[1], index=False, float_format="%.4g")
    return paths


def from_centre_origin(xy, pitch):
    """Convert positions centred on the centre spot (IDSSE, DFL) to metres from a corner."""
    return np.asarray(xy, dtype=float) + np.asarray(pitch, dtype=float) / 2


def _checked(match):
    """Validate a Match's shapes and return (ball, players, team, half, ball_source) as arrays."""
    if match.fps != FPS:
        raise ValueError(f"Feature windows assume {FPS} fps, got {match.fps}: resample to {FPS} fps first")

    ball = np.asarray(match.ball, dtype=float)
    players = np.asarray(match.players, dtype=float)
    team = np.asarray(match.team)
    half = np.asarray(match.half).astype(np.int8)
    n = len(ball)

    if ball.shape != (n, 2) or players.ndim != 3 or players.shape[0] != n or players.shape[2] != 2:
        raise ValueError(f"Expected ball (N, 2) and players (N, S, 2), got {ball.shape} and {players.shape}")
    if team.shape != (players.shape[1],) or not np.isin(team, [HOME, AWAY]).all():
        raise ValueError(f"team must be (S,) = ({players.shape[1]},) of HOME / AWAY codes")
    if half.shape != (n,) or not np.isin(half, [0, 1, 2]).all():
        raise ValueError("half must be (N,) with values 0, 1, 2")
    for name in ("live", "label"):
        value = getattr(match, name)
        if value is not None and np.shape(value) != (n,):
            raise ValueError(f"{name} must be (N,) = ({n},), got {np.shape(value)}")

    # No ball_source (e.g. IDSSE): every frame with a position is observed, the rest missing.
    if match.ball_source is None:
        ball_source = np.where(np.isnan(ball[:, 0]), MISSING, OBSERVED).astype(np.int8)
    else:
        ball_source = np.asarray(match.ball_source)
        if ball_source.shape != (n,):
            raise ValueError(f"ball_source must be (N,) = ({n},), got {ball_source.shape}")
    return ball, players, team, half, ball_source


if __name__ == "__main__":
    # python -m pipeline.feature_table.build_features          (from the repo root)
    table = build_feature_table(load_our_match())
    print(f"{len(table):,} rows x {table.shape[1]} columns")
    for path in save_feature_table(table):
        print("wrote", path)

