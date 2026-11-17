"""Task 3.7: test gap strategies by hiding the ball where it is actually tracked.

No ground truth involved: the answer key is what the real ball says, so it covers the whole match.

    from pipeline.gap_validation import evaluate
    from pipeline.ball_filled import strategy_carry, strategy_midpoint, strategy_signal_a
    evaluate(load_tracking(), {"carry": strategy_carry, "midpoint": strategy_midpoint,
                               "A": strategy_signal_a})

A new strategy (e.g. approach B) only has to follow the shared signature in ball_filled.py.
"""

import warnings

import numpy as np
import pandas as pd

from pipeline import ball_filled as bf
from pipeline.common import AWAY, FPS, HOME, NOBODY, half_of_frame, smooth

# Params
N_BLOCKS = 300
CAP = 30 * FPS             # longest hidden block
MARGIN = FPS               # observed context needed either side of a block (the edge window)
MIN_OBSERVED = 0.95        # share of block + margins where the ball must really be tracked
REFERENCE_SMOOTH = 50      # same majority-vote window as the published floor
PLACEMENT_TRIES = 5000     # random placements tried per drawn block length


def reference_labels(ball, players, team, half):
    """(N,) HOME / AWAY from the real ball: clean frames, carried within each half, smoothed.

    NOBODY outside the graded halves.
    """
    clean = bf.clean_team(ball, players, team).astype(float)
    clean[clean == NOBODY] = np.nan
    carried = pd.Series(clean).groupby(half).transform(lambda s: s.ffill().bfill())
    codes = carried.fillna(NOBODY).to_numpy().astype(np.int8)
    letters = smooth(codes, REFERENCE_SMOOTH)
    out = np.where(letters == "H", HOME, np.where(letters == "A", AWAY, NOBODY)).astype(np.int8)
    out[half == 0] = NOBODY
    return out


def make_blocks(ball, half, gap_lengths, n=N_BLOCKS, seed=0, cap=CAP):
    """(n, 2) disjoint [start, end) blocks where the ball is really tracked, lengths drawn from
    gap_lengths (capped). Each block plus MARGIN either side sits in one graded half with the ball
    observed on at least MIN_OBSERVED of it."""
    rng = np.random.default_rng(seed)
    observed = np.concatenate(([0], np.cumsum(np.isfinite(ball[:, 0]))))
    taken = np.zeros(len(ball), dtype=bool)
    lengths = np.minimum(np.asarray(gap_lengths), cap)
    blocks = []
    for _ in range(3 * n):
        if len(blocks) == n:
            break
        # Draw the length once and retry only its placement: redrawing the length on every
        # failure would let short blocks crowd out long ones, which are harder to place.
        length = int(rng.choice(lengths))
        for _ in range(PLACEMENT_TRIES):
            start = int(rng.integers(MARGIN, len(ball) - length - MARGIN))
            lo, hi = start - MARGIN, start + length + MARGIN
            if half[lo] == 0 or not (half[lo:hi] == half[lo]).all() or taken[lo:hi].any():
                continue
            if (observed[hi] - observed[lo]) / (hi - lo) < MIN_OBSERVED:
                continue
            taken[lo:hi] = True
            blocks.append((start, start + length))
            break
    if len(blocks) < n:
        warnings.warn("make_blocks placed {} of {} blocks".format(len(blocks), n))
    return np.array(sorted(blocks), dtype=int).reshape(-1, 2)


def evaluate(trk, strategies, n=N_BLOCKS, seed=0, half=None, gap_lengths=None):
    """One row per (strategy, bucket): blocks, frame agreement with the reference, median switch
    error in seconds (changed blocks whose reference switches exactly once), and blocks the
    reference switches in more than once (no single-switch strategy can get those right)."""
    half = half_of_frame() if half is None else half
    ball, players, team = trk["ball"], trk["players"], trk["team"]
    if gap_lengths is None:
        starts, ends = bf.long_gaps(ball, half)
        gap_lengths = ends - starts

    reference = reference_labels(ball, players, team, half)
    clean = bf.clean_team(ball, players, team)
    fwd = bf.forward_speed(players, team, half)
    total = len(ball)

    rows = []
    for start, end in make_blocks(ball, half, gap_lengths, n=n, seed=seed):
        s, e = np.array([start]), np.array([end])
        if bf.exit_out(ball, s)[0]:
            continue                                           # likely a stoppage, not this task
        before, after = bf.edge_teams(clean, half, s, e)
        truth = reference[start:end]
        if before[0] == NOBODY or after[0] == NOBODY:
            bucket = "unknown"
        else:
            bucket = "same" if before[0] == after[0] else "changed"
        changes = np.flatnonzero(np.diff(reference[start - 1:end + 1]) != 0)
        for name, strategy in strategies.items():
            pred = strategy(s, e, before, after, fwd, total)[start:end]
            row = {"strategy": name, "bucket": bucket, "multi": len(changes) > 1,
                   "agreement": np.nan if bucket == "unknown" else float(np.mean(pred == truth)),
                   "switch_err_s": np.nan}
            if bucket == "changed" and len(changes) == 1:
                predicted = np.flatnonzero(pred == after[0])
                predicted = predicted[0] if len(predicted) else end - start
                row["switch_err_s"] = abs(int(predicted) - int(changes[0])) / FPS
            rows.append(row)

    table = pd.DataFrame(rows)
    return (table.groupby(["strategy", "bucket"], sort=False)
            .agg(blocks=("agreement", "size"), agreement=("agreement", "mean"),
                 switch_err_s=("switch_err_s", "median"), multi_switch_blocks=("multi", "sum"))
            .reset_index())


def evaluate_decoder(trk, decode_fn, n=N_BLOCKS, seed=0, half=None, gap_lengths=None):
    """Hide the ball on every block at once, decode the whole match once, and score each block.

    decode_fn(hidden_ball) -> (N,) labels "H"/"A"/... . Blocks are disjoint with margins, so hiding
    them together changes nothing at any block's edges. One row per bucket: blocks, mean agreement.
    """
    half = half_of_frame() if half is None else half
    ball, players, team = trk["ball"], trk["players"], trk["team"]
    if gap_lengths is None:
        starts, ends = bf.long_gaps(ball, half)
        gap_lengths = ends - starts

    reference = reference_labels(ball, players, team, half)
    letters = np.where(reference == HOME, "H", np.where(reference == AWAY, "A", "X"))
    clean = bf.clean_team(ball, players, team)
    blocks = [(s, e) for s, e in make_blocks(ball, half, gap_lengths, n=n, seed=seed)
              if not bf.exit_out(ball, np.array([s]))[0]]

    hidden = ball.copy()
    for start, end in blocks:
        hidden[start:end] = np.nan
    labels = decode_fn(hidden)

    rows = []
    for start, end in blocks:
        before, after = bf.edge_teams(clean, half, np.array([start]), np.array([end]))
        if before[0] == NOBODY or after[0] == NOBODY:
            bucket = "unknown"
        else:
            bucket = "same" if before[0] == after[0] else "changed"
        rows.append({"bucket": bucket,
                     "agreement": float(np.mean(labels[start:end] == letters[start:end]))})
    return (pd.DataFrame(rows).groupby("bucket", sort=False)
            .agg(blocks=("agreement", "size"), agreement=("agreement", "mean")).reset_index())
