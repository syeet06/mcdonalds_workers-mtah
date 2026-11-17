"""Step 3: run a trained IDSSE XGBoost model on our match and score it on the public five minutes.

    python -m pipeline.model.predict_our_match                     # both_all7 + tuned Viterbi
    python -m pipeline.model.predict_our_match --raw               # no smoothing, per-frame 0.5 cut

Labels for the whole match, the same way decoder.py writes them:
    X   outside the graded halves
    D   dead ball (Task 4's dead mask) and the second half before its kickoff
    H/A live frames: P(home) (H = HOME = Barcelona, IDSSE label 1) decoded by Viterbi with the
        p_switch from models/smoothing.json (python -m pipeline.model.smoothing), or with --raw
        simply P(home) > 0.5.

Writes evaluation/results/team_xgb_<model>[_raw].csv, validates it and runs the scorer.
The model and p_switch are chosen on IDSSE (steps 1-2 and smoothing.py), not on this score: the
five minutes are the only real test we have.
"""
import json
import argparse

import numpy as np
import pandas as pd
import xgboost as xgb

from pipeline.common import N_FRAMES, SECOND_HALF_PREKICKOFF, half_of_frame, score, submit
from pipeline.feature_table.build_features import build_feature_table, load_our_match
from pipeline.decoder import LETTERS
from pipeline.model.data import feature_columns, load
from pipeline.model.smoothing import SETTINGS as SMOOTHING, decode
from pipeline.model.train_six_tune_seventh import model_path

GT_WINDOW = (11185, 18684)     # video frames of the public five minutes


def our_features():
    """Live frames of our match, columns in training order."""
    table = build_feature_table(load_our_match())
    columns = feature_columns()
    missing = sorted(set(columns) - set(table.columns))
    if missing:
        raise KeyError(f"our feature table lacks training columns: {missing}")
    return table[["frame", "half"]], table[columns].astype(np.float32)


def to_labels(frames, p_home, p_switch=None):
    """(N_FRAMES,) H/A/D/X for the whole match from P(home) on the live frames.

    p_switch None: each live frame on its own (P > 0.5). Otherwise Viterbi within each half, with
    every non-live frame (dead mask, pre-kickoff) forced to D.
    """
    half = half_of_frame()
    first, last = SECOND_HALF_PREKICKOFF
    p = np.full(N_FRAMES, 0.5)
    p[frames] = p_home
    dead = np.ones(N_FRAMES, dtype=bool)
    dead[frames] = False
    dead[first:last + 1] = True

    labels = np.where(half > 0, "D", "X").astype("<U1")
    if p_switch is None:
        live = ~dead & (half > 0)
        labels[live] = np.where(p[live] > 0.5, "H", "A")
        return labels
    for h in (1, 2):
        idx = np.flatnonzero(half == h)
        labels[idx] = LETTERS[decode(p[idx], dead[idx], p_switch)]
    return labels


def compare_inputs(X):
    """How our inputs differ from the degraded IDSSE ones: missing shares, side by side."""
    idsse = load("degraded").X
    rows = {c: (X[c].isna().mean(), idsse[c].isna().mean()) for c in ["ball_x", "d_home", "margin",
            "ball_speed", "frames_since_observed", "n_players_tracked"]}
    print("share missing      ours   IDSSE degraded")
    for c, (ours, theirs) in rows.items():
        print(f"  {c:<22} {ours:6.3f}  {theirs:6.3f}")
    print(f"  ball_source values ours: {sorted(X['ball_source'].dropna().unique().astype(int))}  "
          f"IDSSE: {sorted(idsse['ball_source'].dropna().unique().astype(int))}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", default="xgb_both_all7", help="model name in pipeline/model/models")
    parser.add_argument("--raw", action="store_true", help="no smoothing: each frame P(home) > 0.5")
    parser.add_argument("--compare-inputs", action="store_true", help="print missing shares vs IDSSE")
    args = parser.parse_args()

    model = xgb.XGBClassifier()
    model.load_model(model_path(args.model))
    p_switch = None
    if not args.raw:
        if not SMOOTHING.exists():
            raise FileNotFoundError("no tuned smoothing yet: run python -m pipeline.model.smoothing")
        p_switch = json.loads(SMOOTHING.read_text())["p_switch"]

    print("building our feature table ...")
    meta, X = our_features()
    if args.compare_inputs:
        compare_inputs(X)

    labels = to_labels(meta["frame"].to_numpy(), model.predict_proba(X)[:, 1], p_switch)
    assert len(labels) == N_FRAMES

    window = labels[GT_WINDOW[0]:GT_WINDOW[1] + 1]
    print(f"decoding: {'raw' if p_switch is None else f'viterbi p_switch={p_switch:g}'}; "
          f"five-minute window H/A/D = {(window == 'H').sum()}/{(window == 'A').sum()}/{(window == 'D').sum()}")

    name = "team_xgb_" + args.model.removeprefix("xgb_") + ("_raw" if args.raw else "")
    submit(name, labels)
    score()


if __name__ == "__main__":
    main()
