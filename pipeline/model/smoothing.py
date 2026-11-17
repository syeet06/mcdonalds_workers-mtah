"""Turn XGBoost's per-frame P(home) into possession spells, with the smoothing tuned on IDSSE.

The model is right on most frames but flickers, so it reports 56 possessions where there are 17.
We feed its log-odds into the team's Viterbi decoder (pipeline/decoder.py): every switch between
H and A costs p_switch, so a short contrary blip is ignored and only sustained evidence flips the
label. Entering or leaving a dead ball is free, so a restart can go to either team.

p_switch is chosen here, on IDSSE, never on our five minutes:
  1. Out-of-fold P(home) for every IDSSE match: train on the other six (clean + degraded, the tuned
     settings), predict this one's DEGRADED version.
  2. Decode each match with each p_switch; score against its provider labels with the official
     scorer (evaluation/possession_metrics.py). Dead-ball frames are D on both sides.
  3. Keep the p_switch with the best mean spell-F1 over the seven matches.

    python -m pipeline.model.smoothing            # writes pipeline/model/models/smoothing.json

Rolling-mean windows are scored beside it as a simpler reference.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

from pipeline.decoder import emissions, viterbi
from pipeline.model.compare_clean_degraded import PARAMS
from pipeline.model.data import MATCHES, load
from pipeline.model.train_six_tune_seventh import MODELS, stack

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "evaluation"))
import possession_metrics as pm  # noqa: E402

CACHE = ROOT / "pipeline" / "cache" / "idsse_oof_both.csv.gz"
SETTINGS = MODELS / "smoothing.json"
P_SWITCHES = [0, 1, 2, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96]
WINDOWS = [0, 12, 25, 50, 125]          # rolling-mean half-widths, for comparison
CLIP = 1e-4                             # keep log-odds finite


def log_odds(p):
    p = np.clip(p, CLIP, 1 - CLIP)
    return np.log(p / (1 - p))


def decode(p_home, dead, p_switch):
    """H/A/D codes for one half: P(home) on every frame (any value where dead), dead forced D."""
    score = np.where(dead, 0.0, log_odds(p_home))
    return viterbi(emissions(score, dead), p_switch)   # 0 = H, 1 = A, 2 = D, as in the scorer


def rolling(p_home, dead, half_width):
    """Rolling mean of P(home) over live frames, thresholded at 0.5; dead stays D."""
    p = pd.Series(np.where(dead, np.nan, p_home))
    if half_width:
        p = p.rolling(2 * half_width + 1, center=True, min_periods=1).mean()
    return np.where(dead, pm.D, np.where(p.to_numpy() > 0.5, pm.H, pm.A)).astype(np.int8)


# ------------------------------------------------------------------- out-of-fold predictions ----

def tuned_both():
    """Settings the tuning run chose for the 'both' option (models/settings.json)."""
    chosen = json.loads((MODELS / "settings.json").read_text())["options"]["both"]
    return {**chosen["params"], "n_estimators": chosen["trees"]}


def out_of_fold():
    """Every IDSSE live frame with P(home) from a model that never saw its match. Cached."""
    if CACHE.exists():
        return pd.read_csv(CACHE)
    clean, degraded = load("clean"), load("degraded")
    params = {**PARAMS, **tuned_both()}
    parts = []
    for held_out in MATCHES:
        print(f"  out-of-fold: predicting {held_out}")
        others = [m for m in MATCHES if m != held_out]
        train = stack(clean.select(others), degraded.select(others))
        model = xgb.XGBClassifier(**params).fit(train.X, train.y)
        test = degraded.select(held_out)
        parts.append(test.meta.assign(label=test.y.to_numpy(), p_home=model.predict_proba(test.X)[:, 1]))
    oof = pd.concat(parts, ignore_index=True)
    CACHE.parent.mkdir(exist_ok=True)
    oof.to_csv(CACHE, index=False, float_format="%.5f")
    return oof


def halves(oof):
    """(match_id, gt codes, P(home), dead mask) per match and half, on a continuous frame axis.

    The tables hold live frames only, so a missing frame inside a half is a dead ball.
    """
    for (match_id, _), part in oof.groupby(["match_id", "half"], sort=False):
        frames = part["frame"].to_numpy()
        index = frames - frames.min()
        n = index.max() + 1
        dead = np.ones(n, dtype=bool)
        dead[index] = False
        gt = np.full(n, pm.D, dtype=np.int8)
        gt[index] = np.where(part["label"].to_numpy() == 1, pm.H, pm.A)
        p = np.full(n, 0.5)
        p[index] = part["p_home"].to_numpy()
        yield match_id, gt, p, dead


def score_method(oof, label_fn):
    """Mean over matches of the scorer's metrics, halves of a match joined end to end."""
    per_match = {}
    for match_id, gt, p, dead in halves(oof):
        g, q = per_match.setdefault(match_id, ([], []))
        g.append(gt)
        q.append(label_fn(p, dead))
    rows = []
    for match_id, (g, q) in per_match.items():
        r = pm.evaluate(np.concatenate(g), np.concatenate(q))
        rows.append({k: r[k] for k in ("spell_f1_mean", "macro_f1", "n_gt_spells", "n_pred_spells",
                                       "gt_mean_spell_duration_s", "mean_spell_duration_s")})
    return pd.DataFrame(rows).mean()


def main():
    print("out-of-fold predictions (cached after the first run) ...")
    oof = out_of_fold()

    rows = []
    for w in WINDOWS:
        rows.append({"method": f"rolling ±{w}", **score_method(oof, lambda p, d, w=w: rolling(p, d, w))})
    for s in P_SWITCHES:
        print(f"  viterbi p_switch={s}")
        rows.append({"method": f"viterbi {s}", "p_switch": s,
                     **score_method(oof, lambda p, d, s=s: decode(p, d, s))})
    table = pd.DataFrame(rows)
    print("\nmean over the 7 IDSSE matches (out-of-fold, degraded)")
    print(table.drop(columns="p_switch").round(3).to_string(index=False))

    viterbi_rows = table.dropna(subset=["p_switch"])
    best = viterbi_rows.loc[viterbi_rows["spell_f1_mean"].idxmax()]
    SETTINGS.write_text(json.dumps({"p_switch": float(best["p_switch"]),
                                    "chosen_on": "IDSSE out-of-fold, mean spell_f1_mean",
                                    "spell_f1_mean": round(float(best["spell_f1_mean"]), 4)}, indent=2))
    table.to_csv(MODELS.parent / "results" / "smoothing.csv", index=False, float_format="%.4f")
    print(f"\nchosen p_switch = {best['p_switch']:g} -> {SETTINGS.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
