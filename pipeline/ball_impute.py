"""Infer the ball inside gaps from player movement (gradient boosting).

    from pipeline.ball_impute import infer_real_gaps
    ball_inferred = infer_real_gaps(trk, half, tag_score)      # (N, 2), NaN outside predicted gaps

Each gap frame starts from the straight line between the ball's last and next known positions; two
gradient-boosting models learn a correction (dx, dy) from slot-free player features. Training uses
fake gaps cut into stretches where the tracking has the ball, out of fold by 10-minute chunk.
"""

import warnings

import numpy as np
import pandas as pd

from pipeline.common import AWAY, FPS, HOME

# Params
VEL_LAG = 5                 # frames over which player velocity is measured
SMOOTH = FPS                # team-velocity smoothing, centred
MOVING = 2.0                # m/s; players faster than this count towards the convergence point
NEAREST = 3                 # nearest players per team described around the straight-line guess
CHUNK = 2000                # rows per batch for the pairwise computations


def _nanmean(a, axis):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmean(a, axis=axis)


def _nanstd(a, axis):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanstd(a, axis=axis)


def _gap_rows(ball, half, starts, ends):
    """(frames, line, tau, gap_len_s, edge_dist, p0, p1) for gaps with both edges observed in one half."""
    out = [[] for _ in range(7)]
    n = len(ball)
    for s, e in zip(starts, ends):
        if s < 1 or e >= n or half[s - 1] == 0 or half[s - 1] != half[e]:
            continue
        p0, p1 = ball[s - 1], ball[e]
        if not (np.isfinite(p0).all() and np.isfinite(p1).all()):
            continue
        frames = np.arange(s, e)
        tau = (frames - s + 1) / (e - s + 1)
        out[0].append(frames)
        out[1].append(p0 + tau[:, None] * (p1 - p0))
        out[2].append(tau)
        out[3].append(np.full(len(frames), (e - s) / FPS))
        out[4].append(np.full(len(frames), np.linalg.norm(p1 - p0)))
        out[5].append(np.repeat(p0[None], len(frames), axis=0))
        out[6].append(np.repeat(p1[None], len(frames), axis=0))
    if not out[0]:
        return (np.zeros(0, int), np.zeros((0, 2)), np.zeros(0), np.zeros(0), np.zeros(0),
                np.zeros((0, 2)), np.zeros((0, 2)))
    return tuple(np.concatenate(x) for x in out)


def _team_velocity(players, team):
    """(N, 2, 2) mean velocity per team, smoothed: [frame, team, xy]."""
    vel = np.full(players.shape, np.nan)
    vel[VEL_LAG:] = (players[VEL_LAG:] - players[:-VEL_LAG]) * FPS / VEL_LAG
    out = np.zeros((len(players), 2, 2))
    for t in (HOME, AWAY):
        mean = _nanmean(vel[:, team == t], axis=1)
        out[:, t] = pd.DataFrame(mean).rolling(SMOOTH, center=True, min_periods=1).mean().to_numpy()
    return np.nan_to_num(out), vel


def _convergence(pos, vel):
    """(n, 2) least-squares meeting point of moving players' running directions; NaN if < 3 movers."""
    speed = np.linalg.norm(vel, axis=2)
    ok = np.isfinite(speed) & np.isfinite(pos).all(2) & (speed > MOVING)
    d = np.where(ok[..., None], vel / np.where(speed > 0, speed, 1)[..., None], 0.0)
    w = np.where(ok, speed, 0.0)
    eye = np.eye(2)[None, None]
    m = w[..., None, None] * (eye - d[..., :, None] * d[..., None, :])        # (n, S, 2, 2)
    a = m.sum(1)
    rhs = np.einsum("nsij,nsj->ni", m, np.nan_to_num(pos))
    det = np.linalg.det(a)
    out = np.full((len(pos), 2), np.nan)
    good = (ok.sum(1) >= 3) & (np.abs(det) > 1e-6)
    out[good] = np.linalg.solve(a[good], rhs[good][..., None])[..., 0]
    return out


def gap_features(trk, half, starts, ends, tag_score):
    """One row per frame of every gap with both edges observed. Returns (X, frames, line)."""
    ball, players, team = trk["ball"], trk["players"], trk["team"]
    frames, line, tau, gap_s, edge_d, p0, p1 = _gap_rows(ball, half, starts, ends)
    if len(frames) == 0:
        return pd.DataFrame(), frames, line
    team_vel, vel = _team_velocity(players, team)

    cols = {"tau": tau, "gap_s": gap_s, "edge_dist": edge_d,
            "line_x": line[:, 0], "line_y": line[:, 1], "p0_x": p0[:, 0], "p0_y": p0[:, 1],
            "p1_x": p1[:, 0], "p1_y": p1[:, 1], "tag": tag_score[frames]}
    pos, v = players[frames], vel[frames]
    for t, name in ((HOME, "home"), (AWAY, "away")):
        mine = pos[:, team == t]
        centre = _nanmean(mine, axis=1)
        spread = _nanstd(mine, axis=1)
        cols.update({name + "_cx": centre[:, 0], name + "_cy": centre[:, 1],
                     name + "_sx": spread[:, 0], name + "_sy": spread[:, 1],
                     name + "_vx": team_vel[frames, t, 0], name + "_vy": team_vel[frames, t, 1],
                     name + "_n": np.isfinite(mine[..., 0]).sum(1)})

    conv = _convergence(pos, v)
    centre_all = _nanmean(pos, axis=1)
    conv = np.where(np.isfinite(conv), conv, centre_all)
    cols.update({"conv_dx": conv[:, 0] - line[:, 0], "conv_dy": conv[:, 1] - line[:, 1]})

    press = np.full((len(frames), 3), np.nan)
    near = np.full((len(frames), 2, NEAREST, 3), np.nan)              # [row, team, k, (dx, dy, speed)]
    speed = np.linalg.norm(v, axis=2)
    for lo in range(0, len(frames), CHUNK):
        sl = slice(lo, lo + CHUNK)
        home, away = pos[sl][:, team == HOME], pos[sl][:, team == AWAY]
        pair = np.linalg.norm(home[:, :, None] - away[:, None], axis=3)          # (c, H, A)
        flat = np.nan_to_num(pair, nan=np.inf).reshape(len(pair), -1)
        best = flat.argmin(1)
        hi, ai = np.unravel_index(best, pair.shape[1:])
        r = np.arange(len(pair))
        mid = (home[r, hi] + away[r, ai]) / 2
        press[sl] = np.c_[mid - line[sl], flat[r, best]]
        for t in (HOME, AWAY):
            mine, sp = pos[sl][:, team == t], speed[sl][:, team == t]
            off = mine - line[sl][:, None]
            dist = np.nan_to_num(np.linalg.norm(off, axis=2), nan=np.inf)
            k = min(NEAREST, mine.shape[1])                          # pad when a team has < 3 slots
            order = np.argsort(dist, axis=1)[:, :k]
            near[sl, t, :k, :2] = np.take_along_axis(off, order[..., None], axis=1)
            near[sl, t, :k, 2] = np.take_along_axis(sp, order, axis=1)
            far = np.take_along_axis(dist, order, axis=1) == np.inf
            block = near[sl, t, :k]
            block[far] = np.nan
            near[sl, t, :k] = block
    cols.update({"press_dx": press[:, 0], "press_dy": press[:, 1], "press_dist": press[:, 2]})
    for t, name in ((HOME, "home"), (AWAY, "away")):
        for k in range(NEAREST):
            cols.update({"{}{}_dx".format(name, k): near[:, t, k, 0], "{}{}_dy".format(name, k): near[:, t, k, 1],
                         "{}{}_speed".format(name, k): near[:, t, k, 2]})

    X = pd.DataFrame(cols)
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return X, frames, line


# ------------------------------------------------------------------- training and prediction ----

CHUNK_FRAMES = 10 * 60 * FPS   # 10 minutes of graded play
N_FOLDS = 5
MODEL_PARAMS = dict(random_state=0, max_iter=300, learning_rate=0.05, max_leaf_nodes=31,
                    early_stopping=False)


def chunk_fold(frames, half):
    """Fold 0-4 for each frame: its 10-minute chunk of graded play, round-robin over the folds."""
    offset = np.cumsum(half > 0) - 1                      # graded frames before (and at) each frame
    return (offset[frames] // CHUNK_FRAMES) % N_FOLDS


def fold_splits(folds):
    """(k, train rows, test rows) for each fold present: train never contains fold k."""
    for k in range(N_FOLDS):
        test = np.flatnonzero(folds == k)
        if len(test):
            yield k, np.flatnonzero(folds != k), test


def fit(X, y):
    """Two gradient-boosting models for the (dx, dy) correction to the straight-line guess."""
    from sklearn.ensemble import HistGradientBoostingRegressor

    return tuple(HistGradientBoostingRegressor(**MODEL_PARAMS).fit(X, y[:, i]) for i in (0, 1))


def predict(models, X, line):
    """Straight-line guess plus the predicted correction."""
    return line + np.c_[models[0].predict(X), models[1].predict(X)]


def fake_gaps(trk, half, n=300, seed=0, gap_lengths=None):
    """Hidden-ball blocks on tracked stretches, the same blocks the decoder harness uses."""
    from pipeline.ball_filled import long_gaps
    from pipeline.gap_validation import make_blocks

    if gap_lengths is None:
        starts, ends = long_gaps(trk["ball"], half)
        gap_lengths = ends - starts
    blocks = make_blocks(trk["ball"], half, gap_lengths, n=n, seed=seed)
    return blocks[:, 0], blocks[:, 1]


def _training_rows(trk, half, tag_score, n, seed, gap_lengths):
    starts, ends = fake_gaps(trk, half, n=n, seed=seed, gap_lengths=gap_lengths)
    X, frames, line = gap_features(trk, half, starts, ends, tag_score)
    truth = trk["ball"][frames]
    keep = np.isfinite(truth).all(1)                      # blocks are >= 95% observed, not 100%
    gap_start = starts[np.searchsorted(starts, frames, side="right") - 1]
    return X[keep].reset_index(drop=True), frames[keep], line[keep], truth[keep], gap_start[keep]


def oof_predict(trk, half, tag_score, n=300, seed=0, gap_lengths=None):
    """Out-of-fold predictions on fake gaps: (frames, pred, truth, fold). A gap's fold is that of
    its start frame, and it is only ever predicted by a model trained on the other folds."""
    X, frames, line, truth, gap_start = _training_rows(trk, half, tag_score, n, seed, gap_lengths)
    folds = chunk_fold(gap_start, half)
    pred = np.full_like(truth, np.nan)
    for _, train, test in fold_splits(folds):
        models = fit(X.iloc[train], truth[train] - line[train])
        pred[test] = predict(models, X.iloc[test], line[test])
    return frames, pred, truth, folds


def infer_real_gaps(trk, half, tag_score, n=300, seed=0):
    """(N, 2) inferred ball on real long-gap frames (NaN elsewhere), from a model trained on all fake gaps."""
    from pipeline.ball_filled import long_gaps

    X, _, line, truth, _ = _training_rows(trk, half, tag_score, n, seed, None)
    models = fit(X, truth - line)
    starts, ends = long_gaps(trk["ball"], half)
    Xr, frames, line_r = gap_features(trk, half, starts, ends, tag_score)
    out = np.full(trk["ball"].shape, np.nan)
    if len(frames):
        out[frames] = predict(models, Xr, line_r)
    return out


# ------------------------------------------------------------------------------------ build ----

def build():
    """The ball_impute stage: inferred ball on real long-gap frames, NaN elsewhere."""
    from pipeline import ball_filled as bf
    from pipeline import clues
    from pipeline.common import half_of_frame, load_tracking

    trk = load_tracking()
    half = half_of_frame()
    ball, _ = bf.remove_teleports(trk["ball"])
    ball, _ = bf.fill_short_gaps(ball, half, 25)
    trk = dict(trk, ball=ball)
    intervals = clues.tag_intervals()
    weights = clues.tag_weights(intervals, bf.clean_team(ball, trk["players"], trk["team"]))
    return {"ball_inferred": infer_real_gaps(trk, half, clues.tag_clue(intervals, weights))}
