"""Calculate ball-motion, pitch-position and ball-trust features without loading or saving data.

Feature outputs (one value per frame)
------------------------------------
Feature                    Output         Meaning
C. How the ball moves
ball_speed                 m/s / NaN      Frame-to-frame ball speed, observed frames only
ball_accel                 m/s² / NaN     Change in ball_speed across ~0.2 s (+ speeding up)
ball_turn_angle            degrees / NaN  Change of direction across ~0.2 s, 0 straight to 180 reversal
match_home                 m/s / NaN      |ball_speed - nearest home player's speed|
match_away                 m/s / NaN      |ball_speed - nearest away player's speed|
match_home_away            m/s / NaN      |nearest home player's speed - nearest away player's speed|
slowed_in_home_zone        0 / 1 / NaN    Ball was stopped on this frame, in the half home defends
slowed_in_away_zone        0 / 1 / NaN    Ball was stopped on this frame, in the half away defends
D. Where on the pitch
ball_x                     metres / NaN   Staged ball position along the pitch
ball_y                     metres / NaN   Staged ball position across the pitch
F. How much to trust the ball
ball_source                0 / 1 / 2 / 3  Observed / interpolated / missing / glitch (ball_filled.py)
frames_since_observed      frames / NaN   Frames since the ball was last observed; 0 when observed
n_players_tracked          count          Player slots with coordinates this frame

NaN means unknown. Ball-motion features use observed ball positions only (ball_source == 0):
interpolated positions across long gaps produce speeds of hundreds of m/s and accelerations in the
thousands. Windowed features need every frame in their window observed. Player speeds are measured
from positions over the same window as the other ~0.2 s features, not taken from the ATD.
frames_since_observed resets at half boundaries and is NaN outside playing halves and before the
ball is first observed in a half, like the temporal memory in features_A_B. Divide frame counts by
FPS for seconds.
"""

import numpy as np

from pipeline.ball_filled import MAX_SPEED, OBSERVED, attack_sign
from pipeline.common import AWAY, FPS, HOME, PITCH
from pipeline.dead_ball import STILL_CONFIG

# Params
WINDOW_S = 0.2                    # s: accel, turn angle and player speeds are measured over this window
MIN_MOVE_M = 0.3                  # m: ball must move this far each side of a turn, or the angle is jitter
SLOW_SPEED = STILL_CONFIG["V0"]   # m/s: at or below this the ball counts as slow (Task 4's threshold)
FAST_SPEED = 8.0                  # m/s: the ball must have been at least this fast...
SLOW_LOOK_BACK_S = 0.5            # s: ...within this long before slowing, for a slowdown


def ball_motion_features(ball, ball_source, players, team, half, pitch=PITCH):
    """Return eight named Feature C arrays, each containing one value per frame.

    Inputs: ball (N, 2) and ball_source (N,) from load_stage("ball_filled"); players (N, S, 2)
    and team (S,) from load_tracking(); half (N,) from half_of_frame(). Coordinates in metres
    from a corner; pitch is (length, width), only used to split the pitch into halves.
    """
    players = np.asarray(players, dtype=float)
    n = len(ball)
    observed = ball_source == OBSERVED

    # Frames each side of frame i for a centred WINDOW_S window (2 at 25 fps).
    k = round(WINDOW_S * FPS / 2)
    window_s = 2 * k / FPS

    # Keep only observed positions. Every distance and speed through a NaN stays NaN,
    # so filled or glitched frames can never produce a value.
    obs_ball = np.where(observed[:, None], ball, np.nan)

    # Speed: distance moved since the previous frame, times FPS. Needs both frames observed.
    # Speeds above ball_filled's MAX_SPEED are tracking errors, not kicks.
    ball_speed = np.linalg.norm(obs_ball - _shift(obs_ball, 1), axis=1) * FPS
    ball_speed[ball_speed > MAX_SPEED] = np.nan

    # Acceleration: speed k frames ahead minus speed k frames behind, over the window length.
    # Positive means speeding up (a pass), negative slowing down (a reception).
    ball_accel = (_shift(ball_speed, -k) - _shift(ball_speed, k)) / window_s

    # Turn angle between the incoming (p_i - p_{i-k}) and outgoing (p_{i+k} - p_i) directions.
    # arctan2(cross, dot) is the signed angle; its size is 0 for straight on and 180 for a
    # reversal. It is more accurate than arccos for small angles and needs no division.
    incoming = ball - _shift(ball, k)
    outgoing = _shift(ball, -k) - ball
    dot = (incoming * outgoing).sum(axis=1)
    cross = incoming[:, 0] * outgoing[:, 1] - incoming[:, 1] * outgoing[:, 0]
    angle = np.abs(np.degrees(np.arctan2(cross, dot)))

    # A near-still ball's direction is tracking jitter, so require it to move MIN_MOVE_M each
    # side. Every frame from i-k to i+k must also be observed.
    moving = (np.linalg.norm(incoming, axis=1) > MIN_MOVE_M) & (np.linalg.norm(outgoing, axis=1) > MIN_MOVE_M)
    ball_turn_angle = np.where(_window_all(observed, k) & moving, angle, np.nan)

    # Player speeds from positions over the same window: position k ahead minus k behind.
    # Shape (N, S). A dribbled ball moves at about its carrier's speed, so a small match_*
    # hints that the team's nearest player has the ball.
    player_speed = np.linalg.norm(_shift(players, -k) - _shift(players, k), axis=2) / window_s
    home_speed = _nearest_player_speed(obs_ball, players, player_speed, team, HOME)
    away_speed = _nearest_player_speed(obs_ball, players, player_speed, team, AWAY)

    # Slowdown: the first frame the ball is at or below SLOW_SPEED after being at least
    # FAST_SPEED within the previous SLOW_LOOK_BACK_S. Stack the speeds of the look_back frames
    # before frame i (not including it) and take their maximum, skipping NaNs.
    look_back = round(SLOW_LOOK_BACK_S * FPS)
    past = np.stack([_shift(ball_speed, j) for j in range(1, look_back + 1)])
    has_past = ~np.isnan(past).all(axis=0)
    recent_max = np.full(n, np.nan)
    recent_max[has_past] = np.nanmax(past[:, has_past], axis=0)

    # Missing speeds count as not slow. Only the transition into slow is flagged, so a trapped
    # ball sitting still for 10 frames fires once, not 10 times.
    with np.errstate(invalid="ignore"):
        is_slow = ball_speed <= SLOW_SPEED
        was_fast = recent_max >= FAST_SPEED
    was_slow = np.concatenate([[False], is_slow[:-1]])
    slowdown = is_slow & was_fast & ~was_slow

    # Home zone is the half home defends. Teams swap ends at half time, so attack_sign gives
    # [half, team] = +1 if that team attacks towards +x; then home defends the low-x end.
    sign = attack_sign(players, team, half)
    home_attacks_up = sign[half, HOME] == 1
    with np.errstate(invalid="ignore"):
        in_home_half = np.where(home_attacks_up, obs_ball[:, 0] < pitch[0] / 2, obs_ball[:, 0] > pitch[0] / 2)
    unknown = (half == 0) | np.isnan(ball_speed)

    return {
        "ball_speed": ball_speed,
        "ball_accel": ball_accel,
        "ball_turn_angle": ball_turn_angle,
        "match_home": np.abs(ball_speed - home_speed),
        "match_away": np.abs(ball_speed - away_speed),
        "match_home_away": np.abs(home_speed - away_speed),
        "slowed_in_home_zone": np.where(unknown, np.nan, slowdown & in_home_half),
        "slowed_in_away_zone": np.where(unknown, np.nan, slowdown & ~in_home_half),
    }


def position_features(ball):
    """Return two named Feature D arrays, each containing one value per frame.

    Input: ball (N, 2) from load_stage("ball_filled"), in metres. Interpolated positions are
    included; pair with ball_source to know which positions were observed.
    """
    return {
        "ball_x": ball[:, 0].astype(float),
        "ball_y": ball[:, 1].astype(float),
    }


def trust_features(ball_source, players, half):
    """Return three named Feature F arrays, each containing one value per frame.

    Inputs: ball_source (N,) from load_stage("ball_filled"), players (N, S, 2) from
    load_tracking(), and half (N,) from half_of_frame().
    """
    n = len(ball_source)
    observed = ball_source == OBSERVED

    # Frames since last observed, counted separately in each half so it never runs across
    # halftime. Give each observed frame its own index and others -1; the running maximum then
    # carries the most recent observed index forward through each gap.
    frames_since_observed = np.full(n, np.nan)
    for h in (1, 2):
        in_half = np.flatnonzero(half == h)
        if not len(in_half):
            continue
        last = np.maximum.accumulate(np.where(observed[in_half], in_half, -1))
        # Before the first observation of the half there is nothing to count from: NaN.
        seen = last >= 0
        frames_since_observed[in_half[seen]] = in_half[seen] - last[seen]

    # A slot is tracked when both its x and y are present.
    tracked = ~np.isnan(players).any(axis=2)

    return {
        "ball_source": np.asarray(ball_source),
        "frames_since_observed": frames_since_observed,
        "n_players_tracked": tracked.sum(axis=1),
    }


# ---------------------------------------------------------------------------------- helpers ----

def _shift(values, k):
    """Shift along axis 0 like pandas: positive k looks k frames back, negative looks ahead.

    Frames shifted in from beyond the edges are NaN.
    """
    values = np.asarray(values, dtype=float)
    out = np.full(values.shape, np.nan)
    if k > 0:
        out[k:] = values[:-k]
    elif k < 0:
        out[:k] = values[-k:]
    else:
        out[:] = values
    return out


def _window_all(mask, k):
    """True at frame i only if mask is True on every frame from i-k to i+k."""
    out = np.zeros(len(mask), dtype=bool)
    windows = np.lib.stride_tricks.sliding_window_view(mask, 2 * k + 1)
    out[k:len(mask) - k] = windows.all(axis=1)
    return out


def _nearest_player_speed(ball, players, player_speed, team, side):
    """Speed (m/s) of `side`'s player nearest the ball, per frame. NaN if unknown."""
    slots = team == side
    # Insert a player dimension into ball, (N, 2) -> (N, 1, 2), so it broadcasts against every
    # player of this team. Distances have shape (N, number of team slots).
    d = np.linalg.norm(players[:, slots] - ball[:, None, :], axis=2)

    # Missing positions give NaN distances. Use infinity so argmin never picks them.
    d = np.where(np.isnan(d), np.inf, d)
    nearest = d.argmin(axis=1)
    rows = np.arange(len(d))

    # If every distance is infinite (no ball, or nobody visible), argmin returned an arbitrary
    # player. Those frames are unknown.
    known = np.isfinite(d[rows, nearest])
    return np.where(known, player_speed[:, slots][rows, nearest], np.nan)
