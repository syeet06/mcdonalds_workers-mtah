"""Shared helpers for our pipeline notebooks. Import this instead of re-parsing the data each time.

    import sys; sys.path.insert(0, '..')     # from inside a notebook in pipeline/
    from pipeline.common import *

    trk = load_tracking()                    # ~1 min the first time, instant afterwards (cached)
    trk["ball"]          # (N, 2)  ball x, y in metres, NaN where the ball is missing
    trk["ball_speed"]    # (N,)    ball speed m/s from the ATD, NaN where missing
    trk["players"]       # (N, S, 2) every track slot's x, y in metres, NaN where missing
    trk["player_speed"]  # (N, S)  speed m/s
    trk["team"]          # (S,)    HOME (0) or AWAY (1) for each slot

Row i of every array is video frame i, which is row i of the submission. Labels are numpy arrays of
single characters: "H", "A", "D", "X".

Hand-off between tasks: each task commits pipeline/<stage>.py with a build() function, and
teammates call load_stage("<stage>"), which runs it and caches the result in pipeline/cache/. That
folder is gitignored: it holds match data, which must never reach the public repo. Share code, not data.
"""

import importlib
import os
import subprocess
import time
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "starter"))
from games import GAME, GAMES, half_video_blocks, require  # noqa: E402
from baseline import smooth, team_of_each_track, write_submission  # noqa: E402

# Params
N_FRAMES = GAMES[GAME]["n_video_frames"]  # 199319
FPS = GAMES[GAME]["fps"]  # 25
PITCH = GAMES[GAME]["pitch"]  # (105.3, 68.0) metres
# Graded halves as inclusive (first, last) video frames: (11185, 90651) and (110808, 190197).
HALVES = [(first, last) for _, first, last in half_video_blocks(GAME)]
# Graded, but always D in the ground truth: the second half before its kickoff (CHALLENGE.md).
SECOND_HALF_PREKICKOFF = (110808, 113741)
# Kickoff frames, where each half's match clock starts (00:00 and 45:00). The first half's graded
# block starts on its kickoff; the second's starts SECOND_HALF_PREKICKOFF earlier (117 s).
KICKOFFS = (HALVES[0][0], SECOND_HALF_PREKICKOFF[1] + 1)

HOME, AWAY, NOBODY = 0, 1, -1
CACHE_DIR = os.path.join(HERE, "cache")
BALL_VIDEO_PATH = os.path.join(CACHE_DIR, "ball_video", "ball_video.npz")   # ball_video.build(); here so reading it needs no torch
RESULTS_DIR = os.path.join(ROOT, "evaluation", "results")
# The trained weights the submission loads, committed so nobody has to retrain: the possession model
# (xgb_both_all7.json, from pipeline.model.train_six_tune_seventh --refit) and the fine-tuned ball
# detector (ball_video_finetuned.pt, from pipeline.ball_video_train).
WEIGHTS_DIR = os.path.join(ROOT, "weights")
PUBLIC_GT = os.path.join(ROOT, "evaluation", "gt", "BAR-ATH_possession_gt_public.csv")

# ---------------------------------------------------------------------------------- loading ----

def load_tracking(refresh=False):
    """Tracking as numpy arrays in metres. Parsed from the raw ATD once, then read from cache."""
    cache = os.path.join(CACHE_DIR, "tracking.npz")
    if os.path.exists(cache) and not refresh:
        with np.load(cache) as f:
            return {key: f[key] for key in f.files}

    from pipeline import progress
    progress.note("parsing the raw tracking file (first run only, about a minute); cached afterwards")
    meta_path, raw_path = require(["atd_meta", "atd_raw"])
    team = team_of_each_track(meta_path)
    n_slots = len(team)
    width = 1 + 3 * n_slots + 3  # frame id, (x, y, speed) per slot, (x, y, speed) for the ball

    # Every line has the same shape, so turn the separators into commas and read it as one block.
    with open(raw_path) as handle:
        text = handle.read().replace(":", ",").replace(";", ",").replace("\n", ",")
    values = np.array(text.rstrip(",").split(","), dtype=np.float64).reshape(-1, width)

    frame = values[:, 0].astype(int) - 1  # ATD counts from 1, video frames from 0
    if len(frame) != N_FRAMES or not np.array_equal(frame, np.arange(N_FRAMES)):
        raise ValueError("ATD is not one record per video frame -- see starter/load_atd.py")

    length, pitch_width = PITCH
    slots = values[:, 1:1 + 3 * n_slots].reshape(-1, n_slots, 3)
    ball = values[:, -3:]
    out = {
        "ball": (ball[:, :2] * [length, pitch_width]).astype(np.float32),
        "ball_speed": ball[:, 2].astype(np.float32),
        "players": (slots[:, :, :2] * [length, pitch_width]).astype(np.float32),
        "player_speed": slots[:, :, 2].astype(np.float32),
        "team": team,
    }
    os.makedirs(CACHE_DIR, exist_ok=True)
    np.savez(cache, **out)
    return out


def load_public_gt():
    """Public ground truth as an (N,) label array. X outside the 5-minute window."""
    labels = np.loadtxt(PUBLIC_GT, delimiter=",", skiprows=1, usecols=1, dtype="<U1")
    assert len(labels) == N_FRAMES
    return labels


# ------------------------------------------------------------------------------ frame masks ----

def half_of_frame():
    """(N,) int: 1 or 2 inside a graded half, 0 elsewhere (pre-match, halftime, post-match)."""
    out = np.zeros(N_FRAMES, dtype=np.int8)
    for half, (first, last) in enumerate(HALVES, start=1):
        out[first:last + 1] = half
    return out


def graded_mask():
    """(N,) bool: the frames the scorer grades."""
    return half_of_frame() > 0


# ------------------------------------------------------------------------------- evidence ----

def nearest_team(ball, players, team):
    """(N,) HOME / AWAY / NOBODY: team of the slot closest to the ball. NOBODY if no ball."""
    distance = np.linalg.norm(players - ball[:, None, :], axis=2)  # (N, S), NaN if either missing
    has_answer = ~np.all(np.isnan(distance), axis=1)
    out = np.full(len(ball), NOBODY, dtype=np.int8)
    out[has_answer] = team[np.nanargmin(distance[has_answer], axis=1)]
    return out


# ------------------------------------------------------------------------------- analysing ----

def runs(labels):
    """[(label, start, end_exclusive)] for every maximal run of identical labels."""
    change = np.flatnonzero(labels[1:] != labels[:-1]) + 1
    starts = np.concatenate(([0], change))
    ends = np.concatenate((change, [len(labels)]))
    return [(labels[s], int(s), int(e)) for s, e in zip(starts, ends)]


def summarise(labels, first=0, last=None):
    """Print possession count and mean duration in [first, last]. Compare to GT: 17 in 5 min, 14.2 s."""
    last = N_FRAMES - 1 if last is None else last
    window = labels[first:last + 1]
    spells = [(lab, s, e) for lab, s, e in runs(window) if lab in "HA"]
    deads = [(lab, s, e) for lab, s, e in runs(window) if lab == "D"]
    minutes = (last - first + 1) / FPS / 60
    mean_s = np.mean([e - s for _, s, e in spells]) / FPS if spells else 0.0
    print("{:.1f} min: {} possessions ({:.1f}/min), mean {:.1f} s, {} dead-ball stretches".format(
        minutes, len(spells), len(spells) / minutes, mean_s, len(deads)))


# -------------------------------------------------------------------- hand-off and scoring ----

def save_stage(name, **arrays):
    """Save arrays for teammates, e.g. save_stage("ball_filled", ball=..., ball_source=...)."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    np.savez(os.path.join(CACHE_DIR, name + ".npz"), **arrays)


def load_stage(name):
    """A teammate's output as a dict, e.g. load_stage("ball_filled"). None if it doesn't exist yet.

    The cache is never committed (it is match data), so stages are shared as code instead: each
    task commits pipeline/<name>.py with a build() that returns a dict of arrays. If that file is
    newer than the cached result -- a fresh clone, or a git pull that changed it -- it is re-run
    here and the cache refreshed. Everyone always gets the latest committed version.
    """
    path = os.path.join(CACHE_DIR, name + ".npz")
    module_path = os.path.join(HERE, name + ".py")
    if os.path.exists(module_path) and (
            not os.path.exists(path) or os.path.getmtime(module_path) > os.path.getmtime(path)):
        from pipeline import progress
        progress.note("building {} (first run, or pipeline/{}.py changed); cached afterwards".format(name, name))
        t0 = time.monotonic()
        module = importlib.reload(importlib.import_module("pipeline." + name))
        save_stage(name, **module.build())
        progress.note("built {} in {:.0f} s".format(name, time.monotonic() - t0))
    if not os.path.exists(path):
        return None
    with np.load(path) as f:
        return {key: f[key] for key in f.files}


def submit(name, labels):
    """Write evaluation/results/<name>.csv and run the official validator on it."""
    assert len(labels) == N_FRAMES, "need {} labels, got {}".format(N_FRAMES, len(labels))
    path = write_submission(os.path.join(RESULTS_DIR, name + ".csv"), labels)
    _run([os.path.join("evaluation", "validate_submission.py"), path])
    return path


def score(bootstrap=None):
    """Score every CSV in evaluation/results against the public 5 minutes. Correctness check only."""
    args = [os.path.join("evaluation", "run_evaluation.py"), "--no-detail"]
    if bootstrap:
        args += ["--bootstrap", str(bootstrap)]
    _run(args)


def _run(args):
    result = subprocess.run([sys.executable] + args, cwd=ROOT,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True)
    print(result.stdout)

def match_clock(frame):
    """Frame -> 'mm:ss' on the match clock."""
    h = half_of_frame()[frame]                           # 1, 2, or 0 outside the halves
    if h == 0:
        return '--:--'
    if frame < KICKOFFS[h - 1]:
        return '--:--'                                   # the second half before its kickoff
    secs = (frame - KICKOFFS[h - 1]) / FPS + (45 * 60 if h == 2 else 0)
    return f'{int(secs // 60):02d}:{int(secs % 60):02d}'

