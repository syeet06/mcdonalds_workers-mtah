"""Task 3: clean the ball track and decide who has the ball during ball gaps.

    stage = load_stage("ball_filled")     # runs build() below, cached in pipeline/cache/

Never call load_stage("dead_ball") from here: Task 4 loads this stage, so the two would build each
other forever.
"""

import warnings

import numpy as np
import pandas as pd

from pipeline.common import AWAY, FPS, HOME, NOBODY, PITCH

# Params
MAX_SPEED = 40.0                 # m/s, faster than any shot
MAX_GAP_CANDIDATES = (5, 12, 25, 50)
FILL_ERROR_LIMIT_M = 2.0         # roughly the distance at which a player controls the ball

OBSERVED, INTERPOLATED, MISSING, GLITCH = 0, 1, 2, 3


# -------------------------------------------------------------------------- ball cleaning ----

def remove_teleports(ball, max_speed=MAX_SPEED):
    """Drop single observed points the ball jumps into and straight back out of.

    A lone bad point makes two impossible jumps, one in and one out, so a frame is a glitch only when
    both exceed max_speed. A big jump next to a gap is left alone: it could be a real fast ball.
    Returns (cleaned ball, glitch mask).
    """
    step = np.linalg.norm(np.diff(ball, axis=0), axis=1) * FPS     # step[i]: frame i -> i + 1
    fast = np.nan_to_num(step, nan=0.0) > max_speed
    glitch = np.zeros(len(ball), dtype=bool)
    glitch[1:-1] = fast[:-1] & fast[1:]
    cleaned = ball.copy()
    cleaned[glitch] = np.nan
    return cleaned, glitch


def _gaps(missing):
    """(starts, ends) of runs where `missing` is True, end exclusive."""
    edges = np.diff(np.concatenate(([0], missing.astype(np.int8), [0])))
    return np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)


def fill_short_gaps(ball, half, max_gap):
    """Linear-interpolate gaps of up to max_gap frames with an observed frame on both sides in the
    same graded half. Returns (filled ball, mask of the frames that were filled)."""
    out = ball.copy()
    filled = np.zeros(len(ball), dtype=bool)
    for start, end in zip(*_gaps(np.isnan(ball[:, 0]))):
        if end - start > max_gap or start == 0 or end == len(ball):
            continue
        if half[start - 1] == 0 or half[start - 1] != half[end]:
            continue
        weight = (np.arange(start, end) - (start - 1))[:, None] / (end - (start - 1))
        out[start:end] = ball[start - 1] + weight * (ball[end] - ball[start - 1])
        filled[start:end] = True
    return out, filled


def fill_error_table(ball, half, ks=MAX_GAP_CANDIDATES, n=500, seed=0):
    """Hide k observed frames between two observed ones, interpolate, measure the error in metres.

    Draws n places per k from fully observed stretches inside one graded half.
    """
    rng = np.random.default_rng(seed)
    observed = ~np.isnan(ball[:, 0]) & (half > 0)
    rows = []
    for k in ks:
        span = k + 2
        ok = pd.Series(observed).rolling(span).sum().to_numpy() == span    # window ending at i
        same_half = np.zeros(len(ball), dtype=bool)
        same_half[span - 1:] = half[span - 1:] == half[:len(ball) - span + 1]
        ends = np.flatnonzero(ok & same_half)
        picks = rng.choice(ends, size=min(n, len(ends)), replace=False)
        errors = []
        for last in picks:
            first = last - span + 1
            weight = np.arange(1, k + 1)[:, None] / (k + 1)
            guess = ball[first] + weight * (ball[last] - ball[first])
            errors.append(np.linalg.norm(guess - ball[first + 1:last], axis=1))
        errors = np.concatenate(errors)
        rows.append({"k": k, "median_m": float(np.median(errors)),
                     "p90_m": float(np.percentile(errors, 90))})
    return pd.DataFrame(rows)


def choose_max_gap(table, limit_m=FILL_ERROR_LIMIT_M):
    """Largest k whose 90th-percentile fill error stays under limit_m; 0 if none does."""
    ok = table.loc[table["p90_m"] < limit_m, "k"]
    return int(ok.max()) if len(ok) else 0


# ------------------------------------------------------------------------------ gap table ----

CONTROL_M = 1.5        # nearest player within this of the ball...
MARGIN_M = 2.0         # ...and the nearest opponent at least this much further away
EDGE_WINDOW = FPS      # 1 s either side of a gap
LINE_MARGIN_M = 2.0    # ball this close to a line when it vanished: probably went out


def clean_team(ball, players, team, control=CONTROL_M, margin=MARGIN_M):
    """(N,) HOME / AWAY on frames where one team clearly has the ball at its feet, NOBODY otherwise:
    nearest player within `control` m, nearest opponent at least `margin` m further."""
    distance = np.linalg.norm(players - ball[:, None, :], axis=2)            # (N, S)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)                   # all-NaN frames
        home = np.nanmin(np.where(team == HOME, distance, np.nan), axis=1)
        away = np.nanmin(np.where(team == AWAY, distance, np.nan), axis=1)
    home = np.nan_to_num(home, nan=np.inf)
    away = np.nan_to_num(away, nan=np.inf)
    out = np.full(len(ball), NOBODY, dtype=np.int8)
    with np.errstate(invalid="ignore"):                                    # inf - inf: nobody there
        out[(home <= control) & (away - home >= margin)] = HOME
        out[(away <= control) & (home - away >= margin)] = AWAY
    return out


def long_gaps(ball, half):
    """(starts, ends) of ball gaps inside the graded halves, end exclusive. Halftime splits them."""
    return _gaps(np.isnan(ball[:, 0]) & (half > 0))


def edge_teams(clean, half, starts, ends, window=EDGE_WINDOW):
    """Majority clean team in the `window` frames before and after each gap, same half only.

    NOBODY when there are no clean frames there, or when the two teams tie.
    """
    def vote(lo, hi, which_half):
        lo, hi = max(lo, 0), min(hi, len(clean))
        codes = clean[lo:hi][half[lo:hi] == which_half]
        home, away = np.sum(codes == HOME), np.sum(codes == AWAY)
        return HOME if home > away else AWAY if away > home else NOBODY

    before = np.array([vote(s - window, s, half[s]) for s in starts], dtype=np.int8)
    after = np.array([vote(e, e + window, half[e - 1]) for e in ends], dtype=np.int8)
    return before, after


def exit_out(ball, starts, pitch=None, margin=LINE_MARGIN_M):
    """True where the ball's last position before the gap was within `margin` of a line, or outside."""
    length, width = pitch or PITCH
    last = ball[np.maximum(starts - 1, 0)]
    x, y = last[:, 0], last[:, 1]
    near = (x < margin) | (x > length - margin) | (y < margin) | (y > width - margin)
    return near & np.isfinite(x)


# ------------------------------------------------------------------- signal A and gap_team ----

SMOOTH = 2 * FPS              # forward-speed smoothing, centred
MAX_PLAYER_SPEED = 12.0       # m/s; faster steps are track-slot identity jumps, not running


def attack_sign(players, team, half):
    """(3, 2) array: [half, team] is +1 if that team attacks towards x = PITCH[0] in that half.

    The team with the lower mean x over a half defends the low-x goal, so it attacks towards +x.
    Row 0 (ungraded frames) is all 0.
    """
    sign = np.zeros((3, 2), dtype=np.int8)
    for h in (1, 2):
        frames = half == h
        if not frames.any():
            continue
        mean_x = [np.nanmean(players[frames][:, team == t, 0]) for t in (HOME, AWAY)]
        low = int(np.argmin(mean_x))
        sign[h, low], sign[h, 1 - low] = 1, -1
    return sign


def forward_speed(players, team, half, smooth=SMOOTH):
    """(N, 2) each team's mean speed towards the goal it attacks, m/s, smoothed over `smooth` frames.

    Velocity comes from positions, because the ATD speed has no direction. Frames with nobody
    present contribute nothing to the rolling mean; anything still undefined is 0.
    """
    vx = np.diff(players[:, :, 0], axis=0, prepend=players[:1, :, 0]) * FPS
    vx[np.abs(vx) > MAX_PLAYER_SPEED] = np.nan
    sign = attack_sign(players, team, half)
    out = np.zeros((len(players), 2))
    for t in (HOME, AWAY):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)               # frames with nobody
            raw = np.nanmean(vx[:, team == t], axis=1) * sign[half, t]
        rolled = pd.Series(raw).rolling(smooth, center=True, min_periods=1).mean()
        out[:, t] = rolled.fillna(0.0).to_numpy()
    return out


# Strategies share one signature so the validation harness can swap them:
#     strategy(starts, ends, before, after, fwd, n) -> (n,) int8, NOBODY outside gaps


def _strategy(switch_frame):
    """Wrap a rule that picks the switch frame of a changed gap into a full strategy."""
    def strategy(starts, ends, before, after, fwd, n):
        out = np.full(n, NOBODY, dtype=np.int8)
        for start, end, b, a in zip(starts, ends, before, after):
            if b == NOBODY or a == NOBODY:
                continue
            switch = end if b == a else switch_frame(start, end, b, a, fwd)
            out[start:switch] = b
            out[switch:end] = a
        return out
    return strategy


def _signal_a_switch(start, end, before, after, fwd):
    """Frame where the new team starts pushing forward harder than the old one, nearest the gap's
    middle. No such crossing: the middle of the gap."""
    s = fwd[start:end, after] - fwd[start:end, before]
    ups = np.flatnonzero((s[1:] > 0) & (s[:-1] <= 0)) + 1
    middle = (end - start) // 2
    return start + (int(ups[np.argmin(np.abs(ups - middle))]) if len(ups) else middle)


strategy_carry = _strategy(lambda start, end, b, a, fwd: end)       # Task 1 today
strategy_midpoint = _strategy(lambda start, end, b, a, fwd: start + (end - start) // 2)
strategy_signal_a = _strategy(_signal_a_switch)


# ------------------------------------------------------------------------------------ build ----

def build():
    """The ball_filled stage: cleaned ball, short gaps filled, gap table, and gap_team (signal A)."""
    from pipeline.common import half_of_frame, load_tracking

    trk = load_tracking()
    half = half_of_frame()
    players, team = trk["players"], trk["team"]

    ball, glitch = remove_teleports(trk["ball"])
    max_gap = choose_max_gap(fill_error_table(ball, half))
    ball, filled = fill_short_gaps(ball, half, max_gap)

    ball_source = np.full(len(ball), OBSERVED, dtype=np.int8)
    ball_source[filled] = INTERPOLATED
    ball_source[np.isnan(ball[:, 0])] = MISSING
    ball_source[glitch & ~filled] = GLITCH

    starts, ends = long_gaps(ball, half)
    before, after = edge_teams(clean_team(ball, players, team), half, starts, ends)
    fwd = forward_speed(players, team, half)
    return {
        "ball": ball.astype(np.float32),
        "ball_source": ball_source,
        "gap_team": strategy_signal_a(starts, ends, before, after, fwd, len(ball)),
        "gap_start": starts,
        "gap_end": ends,
        "gap_team_before": before,
        "gap_team_after": after,
        "gap_exit_out": exit_out(ball, starts),
    }
