"""Possession labels from clue scores with a Viterbi decoder.

    from pipeline.decoder import decode
    labels = decode(trk, half_of_frame(), dead=None, tags=(intervals, weights), p_switch=8.0, w_tag=1.0)

Each frame's clue score (+ = HOME, - = AWAY) becomes emission scores for H, A and D; switching
between H and A costs p_switch. So a brief contrary clue is ignored, a sustained one flips the label,
and where the ball is missing the remaining clues (or nothing) decide. D only where it is forced:
Task 4's dead mask, or the second half's pre-kickoff frames.
"""

import numpy as np
import pandas as pd

from pipeline.ball_filled import clean_team
from pipeline.clues import ball_clue, tag_clue, tag_weights
from pipeline.common import AWAY, FPS, HOME, N_FRAMES, SECOND_HALF_PREKICKOFF, runs

# Params
P_SWITCH = 8.0
W_TAG = 1.0
MOVE_WINDOW = 4 * FPS   # how far move_switches may move a switch (IDSSE: gain flat beyond 4 s)
IMPOSSIBLE = -1e9       # finite, so sums never turn into NaN
CARRY_EPS = 1e-6        # tiny lean toward the last team on frames with no evidence
NO_BALL_WHITE_PX = 6    # merge_ballless_spells: fewer white pixels than this in the ball's square...
NO_BALL_SHARE = 0.9     # ...on more than this share of a possession's frames with a square...
NO_BALL_MAX_S = 20      # ...of at most this long...
NO_BALL_MIN_FRAMES = 10   # ...and at least this many frames to judge by

H, A, D = 0, 1, 2
LETTERS = np.array(["H", "A", "D"])


def emissions(score, dead):
    """(T, 3) log-scores: H gets +score/2, A -score/2; D is only possible (and forced) where dead."""
    emit = np.empty((len(score), 3))
    emit[:, H], emit[:, A] = score / 2, -score / 2
    emit[:, D] = IMPOSSIBLE
    emit[dead] = [IMPOSSIBLE, IMPOSSIBLE, 0.0]
    return emit


def viterbi(emit, p_switch):
    """Best state path for (T, 3) emissions. H<->A costs p_switch; entering or leaving D is free.
    Ties go to the lower state index, so a frame with no evidence at all reads as H."""
    trans = np.zeros((3, 3))
    trans[H, A] = trans[A, H] = -p_switch
    n = len(emit)
    back = np.zeros((n, 3), dtype=np.int8)
    best = emit[0].copy()
    for t in range(1, n):
        cand = best[:, None] + trans                     # [from, to]
        back[t] = cand.argmax(axis=0)
        best = cand[back[t], np.arange(3)] + emit[t]
    path = np.empty(n, dtype=np.int8)
    path[-1] = best.argmax()
    for t in range(n - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    return path


def carry_ties(score, half):
    """Give frames with no evidence a tiny lean toward the last team that had some, within each half.

    Without it, every switch point inside a silent stretch scores the same and the tie-break
    always picks HOME, which fills A->H changes early and H->A changes late.
    """
    out = score.astype(float).copy()
    for value, start, end in runs(half):
        if value == 0:
            continue
        seg = out[start:end]
        sign = np.sign(seg)
        sign[sign == 0] = np.nan
        carried = pd.Series(sign).ffill().bfill().fillna(0.0).to_numpy()
        silent = seg == 0
        seg[silent] = CARRY_EPS * carried[silent]
    return out


def decode(trk, half, dead=None, tags=None, p_switch=P_SWITCH, w_tag=W_TAG, extra=None, clue_kw=None):
    """(N,) labels H / A / D, X outside the graded halves. `tags` = (intervals, weights) or None;
    `extra` = any further (N,) clue score, e.g. the inferred-ball clue; `clue_kw` goes to ball_clue."""
    n = len(trk["ball"])
    score = ball_clue(trk["ball"], trk["players"], trk["team"], **(clue_kw or {}))
    if tags is not None and w_tag:
        score = score + w_tag * tag_clue(tags[0], tags[1], n)
    if extra is not None:
        score = score + extra
    score = carry_ties(score, half)
    forced = np.zeros(n, dtype=bool) if dead is None else dead.astype(bool).copy()
    if n == N_FRAMES:
        first, last = SECOND_HALF_PREKICKOFF
        forced[first:last + 1] = True

    labels = np.full(n, "X", dtype="<U1")
    for value, start, end in runs(half):
        if value == 0:
            continue
        path = viterbi(emissions(score[start:end], forced[start:end]), p_switch)
        labels[start:end] = LETTERS[path]
    return labels


def merge_unseen_spells(labels, seen):
    """labels with every possession that has the other team's on both sides, and during which the ball
    was never seen (`seen`: (N,) bool, tracked or video ball), given to that team. Nothing shows the
    team gained the ball, so it is a blip of the model's other evidence (team shape) while the ball is
    hidden. 69819-69842 on our match: Athletic's keeper holds a caught ball, untracked, and a Barcelona
    player picked up again sitting on the goal line swings the shape features to H for 0.9 s.
    Possessions next to a stoppage are never touched. Public window unchanged; that is the only one
    on our match. Tried and dropped: merging
    with up to 10-20% of the spell seen (it merges a real 7 s possession instead), damping the model
    wherever no ball is seen, smoothing the shape features or leaving new tracks out of them
    (each fixes this one and breaks others)."""
    out = labels.copy()
    spans = runs(labels)
    for (before, _, _), (value, s, e), (after, _, _) in zip(spans, spans[1:], spans[2:]):
        if value in "HA" and before in "HA" and before == after and before != value and not seen[s:e].any():
            out[s:e] = before
    return out


def merge_ballless_spells(labels, white):
    """labels with every possession that follows the other team's, lasts at most NO_BALL_MAX_S, and
    on more than NO_BALL_SHARE of its frames shows fewer than NO_BALL_WHITE_PX white pixels in the
    square around the tracked ball (`white`, pipeline/ball_seen.py; -1 = no square, not counted;
    NO_BALL_MIN_FRAMES with a square needed) given to that previous team. The tracking has put the
    ball where there is none, so nothing supports the change of possession. Possessions after a
    stoppage are never touched. Public window unchanged: three possessions (44176, 44819, 56133),
    the same three for any cut-off of 3-10 pixels and a share of 90-95%; at 85% it starts taking real
    ones (132484, 178785)."""
    out = labels.copy()
    spans = runs(labels)
    for (before, _, _), (value, s, e) in zip(spans, spans[1:]):
        if value not in "HA" or before not in "HA" or before == value or e - s > NO_BALL_MAX_S * FPS:
            continue
        w = white[s:e]
        w = w[w >= 0]
        if len(w) >= NO_BALL_MIN_FRAMES and (w < NO_BALL_WHITE_PX).mean() > NO_BALL_SHARE:
            out[s:e] = before
    return out


def move_switches(labels, clean, window=MOVE_WINDOW):
    """Move each direct H<->A switch to the new team's first clean frame after the old team's last.

    The decoder switches once enough evidence has piled up to pay p_switch, so usually late. A loose
    ball stays with the last team, so possession changes when the new team gains it. At most `window`
    frames either way; switches into or out of D stay put; both possessions keep at least one frame.
    On 7 degraded IDSSE matches: spellF1 0.409 -> 0.433,
    better on all 7. Moving to the frame after the old team's last clean frame instead was worse.
    """
    out = labels.copy()
    spans = runs(labels)
    for (old, s1, _), (new, s2, e2) in zip(spans[:-1], spans[1:]):
        if old not in "HA" or new not in "HA" or old == new:
            continue
        old_code, new_code = (HOME, AWAY) if old == "H" else (AWAY, HOME)
        lo, hi = max(s1 + 1, s2 - window), min(e2 - 1, s2 + window)
        old_frames = lo + np.flatnonzero(clean[lo:s2] == old_code)
        base = old_frames[-1] + 1 if len(old_frames) else lo
        new_frames = base + np.flatnonzero(clean[base:hi] == new_code)
        if not len(new_frames):
            continue
        target = new_frames[0]
        if target < s2:
            out[target:s2] = new
        else:
            out[s2:target] = old
    return out


def harness_run(trk, half, intervals, dead=None, p_switch=P_SWITCH, w_tag=W_TAG, extra_fn=None):
    """decode_fn for gap_validation.evaluate_decoder whose tag weights are measured on the hidden
    ball, so the weights never see the frames being tested. `extra_fn(hidden) -> (N,)` adds a clue."""
    def run(hidden):
        weights = tag_weights(intervals, clean_team(hidden, trk["players"], trk["team"]))
        extra = None if extra_fn is None else extra_fn(hidden)
        return decode(dict(trk, ball=hidden), half, dead=dead, tags=(intervals, weights),
                      p_switch=p_switch, w_tag=w_tag, extra=extra)
    return run


def possessions_per_minute(labels, half):
    """Possessions (maximal H or A runs) per minute of graded play."""
    count = 0
    for value, start, end in runs(half):
        if value:
            count += sum(1 for label, _, _ in runs(labels[start:end]) if label in "HA")
    return count / ((half > 0).sum() / FPS / 60)
