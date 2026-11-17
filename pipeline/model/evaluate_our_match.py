"""Apply the XGBoost model + Task 4's dead mask to our match and break down its score on the five minutes.

    python -m pipeline.model.evaluate_our_match            # tuned Viterbi (models/smoothing.json)
    python -m pipeline.model.evaluate_our_match --raw      # no smoothing

Two jobs make up every label, and they are scored separately:
    dead or live?   Task 4's dead mask   -> D, and the live frames handed to the model
    home or away?   the XGBoost model    -> H / A on live frames (Viterbi-smoothed)

All numbers come from the official scorer (evaluation/possession_metrics.py) on the frames the
public ground truth grades. The committed baselines are scored the same way for comparison.
Writes evaluation/results/team_xgb_<model>[_raw].csv and
pipeline/model/results/our_match_evaluation.csv.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

from pipeline.common import N_FRAMES, PUBLIC_GT, RESULTS_DIR
from pipeline.model.predict_our_match import our_features, to_labels
from pipeline.model.smoothing import SETTINGS as SMOOTHING
from pipeline.model.train_six_tune_seventh import model_path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "evaluation"))
import possession_metrics as pm  # noqa: E402

REPORT = Path(__file__).resolve().parent / "results" / "our_match_evaluation.csv"
H, A, D = pm.H, pm.A, pm.D


def predict(model_name, raw):
    """(N_FRAMES,) H/A/D/X labels for our match."""
    model = xgb.XGBClassifier()
    model.load_model(model_path(model_name))
    p_switch = None if raw else json.loads(SMOOTHING.read_text())["p_switch"]
    meta, X = our_features()
    return to_labels(meta["frame"].to_numpy(), model.predict_proba(X)[:, 1], p_switch)


def breakdown(gt, pred):
    """One row of metrics: overall, the dead/live job, and the home/away job."""
    r = pm.evaluate(gt, pred)
    graded = gt != pm.X
    g, p = gt[graded], pred[graded]
    both_live = np.isin(g, [H, A]) & np.isin(p, [H, A])
    pc = r["per_class"]
    return {
        # overall
        "accuracy": r["accuracy"], "macro_f1": r["macro_f1"], "spell_f1": r["spell_f1_mean"],
        "possessions": r["n_pred_spells"], "gt_possessions": r["n_gt_spells"],
        # dead vs live (Task 4's mask)
        "dead_precision": pc["D"]["precision"], "dead_recall": pc["D"]["recall"], "dead_f1": pc["D"]["f1"],
        "live_dead_f1": r["alive_dead_f1"],
        "dead_called_live": int(((g == D) & (p != D)).sum()),
        "live_called_dead": int((np.isin(g, [H, A]) & (p == D)).sum()),
        # home vs away (the model), on frames both sides call live
        "team_accuracy": (g[both_live] == p[both_live]).mean(),
        "home_precision": pc["H"]["precision"], "home_recall": pc["H"]["recall"],
        "away_precision": pc["A"]["precision"], "away_recall": pc["A"]["recall"],
        "n_graded": int(graded.sum()),
    }


def confusion_table(gt, pred):
    graded = gt != pm.X
    cm = pm.confusion(gt[graded], pred[graded])[:3, :3]
    return pd.DataFrame(cm, index=["truth H", "truth A", "truth D"], columns=["pred H", "pred A", "pred D"])


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", default="xgb_both_all7", help="model name in pipeline/model/models")
    parser.add_argument("--raw", action="store_true", help="no smoothing: each frame P(home) > 0.5")
    args = parser.parse_args()

    print("predicting our match ...")
    labels = predict(args.model, args.raw)
    name = "team_xgb_" + args.model.removeprefix("xgb_") + ("_raw" if args.raw else "")
    path = Path(RESULTS_DIR) / f"{name}.csv"
    submission = pd.DataFrame({"frame": np.arange(N_FRAMES), "label": labels})
    problem = pm.validate_submission(submission, N_FRAMES)
    if problem:
        raise ValueError(f"invalid submission: {problem}")
    submission.to_csv(path, index=False)

    gt = pm.label_codes(pm.read_label_csv(PUBLIC_GT))
    ours = pm.label_codes(labels)
    rows = {name: breakdown(gt, ours)}
    for baseline in sorted(Path(RESULTS_DIR).glob("baseline_*.csv")):
        rows[baseline.stem] = breakdown(gt, pm.label_codes(pm.read_label_csv(baseline)))
    table = pd.DataFrame(rows).T

    print(f"\n{name}: {rows[name]['n_graded']} graded frames (the public five minutes)\n")
    print("confusion matrix (frames)")
    print(confusion_table(gt, ours).to_string())

    sections = {
        "overall": ["accuracy", "macro_f1", "spell_f1", "possessions", "gt_possessions"],
        "dead vs live  (Task 4's dead mask)": ["dead_precision", "dead_recall", "dead_f1", "live_dead_f1",
                                               "dead_called_live", "live_called_dead"],
        "home vs away  (XGBoost; team_accuracy on frames both call live, H/A precision/recall on all graded)": ["team_accuracy", "home_precision", "home_recall",
                                                          "away_precision", "away_recall"],
    }
    for title, columns in sections.items():
        print(f"\n{title}")
        print(table[columns].astype(float).round(3).to_string())

    REPORT.parent.mkdir(exist_ok=True)
    table.to_csv(REPORT, index_label="submission", float_format="%.4f")
    print(f"\nwrote {path.relative_to(ROOT)} and {REPORT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
