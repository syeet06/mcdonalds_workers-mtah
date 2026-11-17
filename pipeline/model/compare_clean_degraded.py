"""Step 1: does training on degraded IDSSE beat training on clean IDSSE?

For each rotation, train two XGBoost models with identical settings on the same two matches -- one
on the clean table, one on the degraded table -- and score both on a third, held-out match, in both
its clean and degraded versions. The degraded score is the one that matters: it is what our match
looks like. The clean score is there to show what each model gives up.

    python3 -m pipeline.model.compare_clean_degraded                       # 3 default rotations
    python3 -m pipeline.model.compare_clean_degraded --train J03WMX J03WN1 --test J03WPY

Writes pipeline/model/results/compare_clean_degraded.csv (one row per rotation x model x test set).
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score

from pipeline.model.data import MATCHES, check_aligned, load

RESULTS = Path(__file__).resolve().parent / "results" / "compare_clean_degraded.csv"

# Fixed, untuned settings: the comparison is clean vs degraded data, not hyperparameters.
# No early stopping -- there is no validation match here, and stopping on the test match would leak.
PARAMS = dict(n_estimators=400, learning_rate=0.05, max_depth=6, min_child_weight=5,
              subsample=0.8, colsample_bytree=0.8, tree_method="hist",
              objective="binary:logistic", eval_metric="logloss", n_jobs=-1, random_state=0)

# Rolling-mean window (frames each side) for the smoothed score. Our submission will be smoothed,
# and a model that flickers can look fine per frame and fall apart once smoothed.
SMOOTH = 25

# Train on two matches, test on a third; no match is tested twice.
ROTATIONS = [(MATCHES[0:2], MATCHES[2]), (MATCHES[3:5], MATCHES[5]), ([MATCHES[6], MATCHES[0]], MATCHES[1])]


def smooth(p, meta, half_width=SMOOTH):
    """Centred rolling mean of p within each match and half, in frame order."""
    s = pd.Series(p, index=meta.index)
    order = meta.sort_values(["match_id", "half", "frame"]).index
    rolled = (s.loc[order].groupby([meta.loc[order, "match_id"], meta.loc[order, "half"]])
              .transform(lambda v: v.rolling(2 * half_width + 1, center=True, min_periods=1).mean()))
    return rolled.loc[meta.index].to_numpy()


def score(model, test):
    """Per-frame metrics on one test set, overall and split by whether the ball is observed."""
    p = model.predict_proba(test.X)[:, 1]
    y = test.y.to_numpy()
    ball_missing = test.X["ball_x"].isna().to_numpy()
    acc = lambda mask: accuracy_score(y[mask], p[mask] > 0.5) if mask.any() else np.nan
    return {
        "n": len(y),
        "accuracy": acc(np.ones_like(y, dtype=bool)),
        "log_loss": log_loss(y, np.clip(p, 1e-6, 1 - 1e-6)),
        "auc": roc_auc_score(y, p),
        "acc_ball_seen": acc(~ball_missing),
        "acc_ball_missing": acc(ball_missing),
        "share_ball_missing": ball_missing.mean(),
        f"acc_smooth{SMOOTH}": accuracy_score(y, smooth(p, test.meta) > 0.5),
    }


def run(rotations):
    print("loading tables ...")
    tables = {"clean": load("clean"), "degraded": load("degraded")}
    check_aligned(tables["clean"], tables["degraded"])

    rows = []
    for train_ids, test_id in rotations:
        print(f"\ntrain {' + '.join(train_ids)}  ->  test {test_id}")
        for trained_on, table in tables.items():
            train, _ = table.split(train_ids, [test_id])
            model = xgb.XGBClassifier(**PARAMS).fit(train.X, train.y)
            for tested_on, test_table in tables.items():
                result = score(model, test_table.select(test_id))
                rows.append({"train": "+".join(train_ids), "test": test_id,
                             "trained_on": trained_on, "tested_on": tested_on, **result})
                print(f"  trained {trained_on:<8} tested {tested_on:<8} "
                      f"acc {result['accuracy']:.3f}  logloss {result['log_loss']:.3f}  "
                      f"auc {result['auc']:.3f}  ball-missing acc {result['acc_ball_missing']:.3f}")
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--train", nargs=2, metavar="MATCH", help="two training match ids")
    parser.add_argument("--test", metavar="MATCH", help="held-out match id")
    args = parser.parse_args()
    if bool(args.train) != bool(args.test):
        parser.error("give --train and --test together, or neither for the default rotations")
    rotations = [(args.train, args.test)] if args.train else ROTATIONS

    results = run(rotations)
    RESULTS.parent.mkdir(exist_ok=True)
    results.to_csv(RESULTS, index=False, float_format="%.4f")

    metrics = ["accuracy", "log_loss", "auc", "acc_ball_seen", "acc_ball_missing", f"acc_smooth{SMOOTH}"]
    print("\nmean over rotations (rows: trained on, tested on)")
    print(results.groupby(["tested_on", "trained_on"])[metrics].mean().round(3).to_string())
    print(f"\nwrote {RESULTS.relative_to(Path.cwd()) if RESULTS.is_relative_to(Path.cwd()) else RESULTS}")
    print("Decide on the 'tested_on = degraded' rows: that is what our match looks like.")


if __name__ == "__main__":
    main()
