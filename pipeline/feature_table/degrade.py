"""Make clean external tracking (IDSSE) look like our ATD before building its feature table.

    match = degrade_match(idsse_match, seed=seed_for(match_id))
    table = build_feature_table(match)

IDSSE sees the ball and every player on almost every frame; we don't. A model trained on the clean
data learns to trust a ball that is always there. degrade_match only blanks positions (never moves
them), in the Match, so every feature group (memory, speeds, time shifts) sees the gaps the same
way it does on our match. Labels, live mask and halves are untouched: the degraded table has the
same rows as the clean one.

The targets are our match's live frames, measured with degradation_stats() below:

Ball     26.7 % of live frames unobserved, in long gaps (median 6.8 s, max 56 s). A gap starts
         more often in a duel (x2.4), when the ball is slow, and in the last 17.5 m at each end.
Players  21.1 of 22 tracked on average. Single-player dropouts last a median 2.4 s; 8.2 % of
         frames have < 20 tracked; ~1 % have almost everyone gone; 0.6 % a whole team gone.

Not modelled (v1): ghost slots (> 11 on a team), track identity breaks, ball glitches.
"""
import dataclasses
import zlib

import numpy as np
import pandas as pd

from pipeline.ball_filled import MISSING, OBSERVED
from pipeline.common import AWAY, FPS, HOME
from pipeline.feature_table.features_A_B import distance_features

# Ball gaps: our gap lengths in frames, (quantile, length); sampled in log space between points.
GAP_QUANTILES = ((0.0, 1), (0.1, 5), (0.25, 44), (0.5, 169), (0.75, 284), (0.9, 628), (0.99, 1117), (1.0, 1391))
BALL_UNOBSERVED = 0.267       # share of live frames with no observed ball
DUEL_FACTOR = 2.4             # gap-start weight in a duel (both teams within 1.5 m)...
SLOW_FACTOR = 2.0             # ...when the ball is slower than SLOW_MS...
END_FACTOR = 3.0              # ...and in the last END_M metres at either end
SLOW_MS = 3.0
END_M = 17.5
DUEL_M = 1.5

BALL_SEPARATION = FPS         # frames kept between gaps, so clustered starts don't fuse gaps

# Players: independent single-player dropouts, plus rarer correlated events. Rates are set so the
# totals land on ours: mean tracked, share of frames under 20, share with a whole team gone.
MEAN_TRACKED = 21.08
PLAYER_MISSING = 0.017        # share of a player's live frames lost to single dropouts
PLAYER_GAP = (60, 1.65)       # lognormal (median frames, sigma): median 2.4 s, p90 ~20 s
GROUP = 0.060                 # share of live frames where 3-8 players drop together
GROUP_GAP = (60, 1.0)
NEAR_ALL = 0.010              # share of live frames where only 1-3 players per team are left
NEAR_ALL_GAP = (50, 1.0)
TEAM_GONE = 0.001             # share of live frames where one whole team is gone (near-all adds ~0.5 %)
TEAM_GONE_GAP = (166, 0.5)
MAX_GAP = 2500                # frames, cap for every sampled burst


def seed_for(match_id):
    """Stable seed from a match id, so a rebuild degrades the same frames."""
    return zlib.crc32(str(match_id).encode())


def degrade_match(match, seed):
    """Return a copy of `match` with ball and player positions blanked to look like our ATD."""
    rng = np.random.default_rng(seed)
    half = np.asarray(match.half)
    live = (half > 0) if match.live is None else (np.asarray(match.live, bool) & (half > 0))
    segment_end = _segment_end(half)
    ball = np.array(match.ball, dtype=float)
    players = np.array(match.players, dtype=float)
    team = np.asarray(match.team)

    # Ball: long gaps, starting where ours tend to start, until the live share matches ours.
    speed = _ball_speed(ball, half, match.fps)
    # Our gap lengths were measured inside stretches of live play, so a gap must fit in one:
    # crossing a dead ball would cut it in two and make gaps shorter and more frequent than ours.
    weights = ball_start_weights(ball, speed, players, team, match.pitch) * live
    live_run = np.cumsum(np.r_[True, (live[1:] != live[:-1]) | (half[1:] != half[:-1])])
    gone = _bursts(weights, _segment_end(live_run), _sample_gap, BALL_UNOBSERVED, live, rng,
                   already=np.isnan(ball[:, 0]), separation=BALL_SEPARATION, fit=True)
    ball[gone] = np.nan
    if match.ball_source is None:
        source = np.where(np.isnan(match.ball[:, 0]), MISSING, OBSERVED).astype(np.int8)
    else:
        source = np.array(match.ball_source, dtype=np.int8)
    source[gone] = MISSING

    # Players: each one drops out on its own; then groups, near-total and whole-team blackouts.
    for s in range(players.shape[1]):
        present = live & np.isfinite(players[:, s, 0])
        if present.any():
            lost = _bursts(present.astype(float), segment_end, _lognormal(*PLAYER_GAP), PLAYER_MISSING, present, rng)
            players[lost, s] = np.nan
    on = live.astype(float)
    for start, end in _runs(_bursts(on, segment_end, _lognormal(*GROUP_GAP), GROUP, live, rng)):
        here = np.flatnonzero(np.isfinite(players[start, :, 0]))     # players on the pitch now
        players[start:end, _pick(rng, here, rng.integers(3, 9))] = np.nan
    for start, end in _runs(_bursts(on, segment_end, _lognormal(*NEAR_ALL_GAP), NEAR_ALL, live, rng)):
        here = np.isfinite(players[start, :, 0])
        keep = np.concatenate([_pick(rng, np.flatnonzero(here & (team == t)), rng.integers(1, 4)) for t in (HOME, AWAY)])
        drop = np.setdiff1d(np.arange(players.shape[1]), keep)
        players[start:end, drop] = np.nan
    for start, end in _runs(_bursts(on, segment_end, _lognormal(*TEAM_GONE_GAP), TEAM_GONE, live, rng)):
        players[start:end, team == rng.choice([HOME, AWAY])] = np.nan

    return dataclasses.replace(match, ball=ball, players=players, ball_source=source)


def ball_start_weights(ball, speed, players, team, pitch):
    """(N,) relative chance a ball gap starts on each frame: duel x slow ball x pitch end."""
    a = distance_features(ball, players, team)
    with np.errstate(invalid="ignore"):
        duel = (a["d_home"] <= DUEL_M) & (a["d_away"] <= DUEL_M)
        slow = speed < SLOW_MS
        end = np.abs(ball[:, 0] - pitch[0] / 2) > pitch[0] / 2 - END_M
    w = np.where(duel, DUEL_FACTOR, 1.0) * np.where(slow, SLOW_FACTOR, 1.0) * np.where(end, END_FACTOR, 1.0)
    return np.where(np.isfinite(ball[:, 0]), w, 0.0)


def _ball_speed(ball, half, fps):
    """(N,) ball speed in m/s from positions, smoothed over 5 frames; NaN across halves or gaps."""
    step = np.linalg.norm(np.diff(ball, axis=0, prepend=ball[:1]), axis=1) * fps
    step[np.r_[True, half[1:] != half[:-1]]] = np.nan
    return pd.Series(step).rolling(5, center=True, min_periods=1).mean().to_numpy()


def _bursts(weights, segment_end, sample_len, target, denominator, rng, already=None, separation=0,
            fit=False, patience=200):
    """Bool mask of new bursts, added one at a time until (already | new) covers `target` of
    `denominator` frames. Starts are drawn in proportion to `weights`; each burst is cut at its
    segment's end, or with `fit` must end inside its segment. With `separation`, a burst that
    would come within that many frames of a covered frame is moved. Returns only the new frames."""
    n = len(weights)
    covered = np.zeros(n, bool) if already is None else np.asarray(already, bool).copy()
    new = np.zeros(n, bool)
    total = weights.sum()
    if total <= 0 or not denominator.any():
        return new
    p = weights / total
    need = int(round(target * denominator.sum()))
    stale = 0
    while (short := need - covered[denominator].sum()) > 0 and stale < patience:
        length = int(sample_len(rng, 1)[0])
        # Keep the length and move the start (redrawing the length would favour short bursts).
        # With `fit`, draw only from starts that leave room; no room anywhere: the longest that fits.
        q = p
        if fit:
            room = segment_end - np.arange(n)
            length = min(length, int(room[weights > 0].max()))
            q = np.where(room >= length, weights, 0.0)
            q = q / q.sum()
        for start in rng.choice(n, size=50, p=q):
            end = min(start + length, segment_end[start])
            if not (separation and covered[max(0, start - separation):end + separation].any()):
                break
        else:
            stale += 1
            continue
        # Cut the burst that would overshoot, so short matches land on the target too.
        adds = np.cumsum(~covered[start:end] & denominator[start:end])
        if adds.size and adds[-1] > short:
            end = start + int(np.searchsorted(adds, short)) + 1
        stale = 0 if (~covered[start:end]).any() else stale + 1
        covered[start:end] = True
        new[start:end] = True
    return new


def _sample_gap(rng, k):
    """k ball-gap lengths (frames) from our empirical quantiles."""
    q, length = np.array(GAP_QUANTILES, dtype=float).T
    return np.maximum(1, np.round(np.exp(np.interp(rng.uniform(size=k), q, np.log(length))))).astype(int)


def _lognormal(median, sigma):
    """Length sampler: lognormal frames with this median and sigma, capped at MAX_GAP."""
    return lambda rng, k: np.clip(np.round(rng.lognormal(np.log(median), sigma, k)), 1, MAX_GAP).astype(int)


def _segment_end(half):
    """(N,) index one past the end of each frame's half-run (halves sit back to back in IDSSE)."""
    half = np.asarray(half)
    change = np.flatnonzero(np.r_[half[1:] != half[:-1], True]) + 1
    return change[np.searchsorted(change, np.arange(len(half)), side="right")]


def _pick(rng, slots, k):
    """Up to k distinct slots from `slots`, at random."""
    return rng.choice(slots, size=min(len(slots), int(k)), replace=False)


def _runs(mask):
    """(start, end) of each run of True, end exclusive."""
    edges = np.diff(np.r_[0, mask.astype(np.int8), 0])
    return zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1))


# ------------------------------------------------------------------- check it worked ----

def degradation_stats(table, fps=FPS):
    """The degradation statistics above (the targets) for any live-frame feature table.

    Gaps are runs of frames without an observed ball, split at frame jumps (dead ball), half-time
    and match boundaries (`match_id`, if present). Put degraded IDSSE next to our match.
    """
    match = table["match_id"].to_numpy() if "match_id" in table else np.zeros(len(table))
    frame = table["frame"].to_numpy()
    half = table["half"].to_numpy()
    unobs = (table["ball_source"] != OBSERVED).to_numpy()
    block = np.r_[True, (np.diff(frame) != 1) | (half[1:] != half[:-1]) | (match[1:] != match[:-1])]
    starts = np.flatnonzero(unobs & (block | np.r_[True, ~unobs[:-1]]))
    ends = np.flatnonzero(unobs & np.r_[block[1:] | ~unobs[1:], True]) + 1
    lengths = pd.Series(ends - starts, dtype=float)
    has_before = ~block[starts]                            # the frame before is the same stretch
    duel = table["is_duel"].to_numpy()
    observed = ~unobs
    return {
        "pct_live_frames_ball_unobserved": round(100 * unobs.mean(), 1),
        "n_gaps": len(lengths),
        "ball_gaps_per_live_minute": round(len(lengths) / (len(table) / fps / 60), 2),
        "gap_length_frames_quantiles": lengths.quantile([.1, .25, .5, .75, .9, .99]).round().to_dict(),
        "gap_length_frames_max": int(lengths.max()) if len(lengths) else 0,
        "pct_unobserved_in_gaps_over_10s": round(100 * lengths[lengths > 10 * fps].sum() / max(lengths.sum(), 1), 1),
        "p_duel_before_gap": round(float(np.nanmean(duel[starts[has_before] - 1])), 3) if has_before.any() else None,
        "p_duel_observed": round(float(np.nanmean(duel[observed])), 3) if observed.any() else None,
        "mean_players_tracked": round(float(table["n_players_tracked"].mean()), 2),
        "pct_frames_lt20_players": round(100 * (table["n_players_tracked"] < 20).mean(), 1),
        "pct_frames_team_untracked": round(100 * table["centroid_x_diff"].isna().mean(), 2),
    }
