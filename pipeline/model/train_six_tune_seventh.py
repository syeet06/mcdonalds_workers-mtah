"""Step 2: train on six IDSSE matches, tune on the seventh, save the models for our match.

Three training sets, all on the same six matches:
    clean     the clean table
    degraded  the degraded table
    both      clean and degraded rows stacked (twice the rows, same labels)

The seventh match is the validation match. It is always scored in its DEGRADED version, because
that is what our match looks like. It picks the number of trees (early stopping) and, with --tune,
the tree settings from a small grid. Our own five minutes are never used here.

    python -m pipeline.model.train_six_tune_seventh                      # validate on J03WR9
    python -m pipeline.model.train_six_tune_seventh --val J03WMX --tune
    python -m pipeline.model.train_six_tune_seventh --refit              # also refit on all 7

Saves pipeline/model/models/xgb_<option>.json (six-match model, early-stopped) and, with --refit,
xgb_<option>_all7.json (all seven matches, same settings and tree count scaled up by 7/6).
Results: pipeline/model/results/train_six_tune_seventh.csv.
"""
import argparse
import itertools
import json
from pathlib import Path

import pandas as pd
import xgboost as xgb

from pipeline.model.compare_clean_degraded import PARAMS, score
from pipeline.model.data import MATCHES, Dataset, check_aligned, load

HERE = Path(__file__).resolve().parent
MODELS = HERE / "models"


def model_path(name):
    """models/<name>.json if trained here, else the committed copy in weights/ (only xgb_both_all7)."""
    from pipeline.common import WEIGHTS_DIR
    local = MODELS / f"{name}.json"
    return local if local.exists() else Path(WEIGHTS_DIR) / f"{name}.json"
RESULTS = HERE / "results" / "train_six_tune_seventh.csv"

OPTIONS = ("clean", "degraded", "both")
MAX_TREES = 3000
EARLY_STOP = 100            # stop after this many trees without a better validation log-loss
GRID = {"max_depth": [4, 6, 8], "min_child_weight": [1, 10]}


def stack(a, b):
    """Rows of two aligned Datasets one after the other (the 'both' option)."""
    return Dataset("both", pd.concat([a.X, b.X], ignore_index=True),
                   pd.concat([a.y, b.y], ignore_index=True),
                   pd.concat([a.meta, b.meta], ignore_index=True))


def fit(train, val, params):
    model = xgb.XGBClassifier(**{**PARAMS, **params, "n_estimators": MAX_TREES,
                                 "early_stopping_rounds": EARLY_STOP})
    return model.fit(train.X, train.y, eval_set=[(val.X, val.y)], verbose=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--val", default=MATCHES[-1], choices=MATCHES, help="validation match")
    parser.add_argument("--tune", action="store_true", help=f"grid-search {GRID} on the validation match")
    parser.add_argument("--refit", action="store_true", help="also refit each option on all seven matches")
    args = parser.parse_args()

    print("loading tables ...")
    clean, degraded = load("clean"), load("degraded")
    check_aligned(clean, degraded)
    train_ids = [m for m in MATCHES if m != args.val]
    print(f"train {', '.join(train_ids)}\nvalidate {args.val} (degraded)")

    val = {"clean": clean.select(args.val), "degraded": degraded.select(args.val)}
    train = {"clean": clean.select(train_ids), "degraded": degraded.select(train_ids)}
    train["both"] = stack(train["clean"], train["degraded"])

    grid = [dict(zip(GRID, values)) for values in itertools.product(*GRID.values())] if args.tune else [{}]
    MODELS.mkdir(exist_ok=True)
    rows, chosen = [], {}
    for option in OPTIONS:
        best = None
        for params in grid:
            model = fit(train[option], val["degraded"], params)
            result = score(model, val["degraded"])
            label = ", ".join(f"{k}={v}" for k, v in params.items()) or "default settings"
            print(f"  {option:<8} {label:<34} trees {model.best_iteration + 1:>4}  "
                  f"val logloss {result['log_loss']:.4f}  acc {result['accuracy']:.3f}")
            if best is None or result["log_loss"] < best[1]["log_loss"]:
                best = (model, result, params)

        model, result, params = best
        clean_result = score(model, val["clean"])
        trees = model.best_iteration + 1
        model.save_model(MODELS / f"xgb_{option}.json")
        chosen[option] = {"params": {**PARAMS, **params}, "trees": trees}
        rows.append({"option": option, "val": args.val, "trees": trees, **params,
                     **{f"deg_{k}": v for k, v in result.items()},
                     **{f"clean_{k}": v for k, v in clean_result.items()}})

        if args.refit:
            everything = {"clean": clean, "degraded": degraded, "both": stack(clean, degraded)}[option]
            final = xgb.XGBClassifier(**{**PARAMS, **params, "n_estimators": round(trees * 7 / 6)})
            final.fit(everything.X, everything.y).save_model(MODELS / f"xgb_{option}_all7.json")

    (MODELS / "settings.json").write_text(json.dumps({"val": args.val, "train": train_ids,
                                                      "options": chosen}, indent=2, default=str))
    results = pd.DataFrame(rows)
    RESULTS.parent.mkdir(exist_ok=True)
    results.to_csv(RESULTS, index=False, float_format="%.4f")

    print(f"\nvalidation match {args.val}, best settings per option")
    shown = ["option", "trees", *GRID] if args.tune else ["option", "trees"]
    shown += ["deg_accuracy", "deg_log_loss", "deg_auc", "deg_acc_ball_missing", "clean_accuracy"]
    print(results[shown].round(4).to_string(index=False))
    print(f"\nmodels in {MODELS.relative_to(Path.cwd()) if MODELS.is_relative_to(Path.cwd()) else MODELS}")
    print("Pick the option on deg_log_loss / deg_accuracy. One validation match is noisy: "
          "rerun with another --val before trusting a gap under ~0.01.")


if __name__ == "__main__":
    main()
