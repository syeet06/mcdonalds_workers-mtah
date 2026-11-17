# XGBoost possession model

We train XGBoost on seven public matches (IDSSE), then use it to label our match. On the public
five minutes it is **first on the board: spell-F1 0.594, macro-F1 0.881**, against 0.424 and
0.621 for the best baselines. It finds 18 possessions; the truth has 17.

## Rebuild the models (after a fresh clone or pull)

The trained models in `pipeline/model/models/` are gitignored (~30 MB), so `git pull` does not
bring them. The one the submission loads is committed separately, as `weights/xgb_both_all7.json`. Rebuild them by running these from the repo root.

1. Set up Python 3.14 (Python 3.15 can't install the pinned scipy). Use whichever command starts
   Python 3.14 on your machine:

   ```bash
   python3.14 -m venv .venv
   source .venv/bin/activate
   pip install -r pipeline/requirements.txt
   ```

2. Build the IDSSE training tables, if `pipeline/feature_table/idsse_all_matches_features*.csv.gz`
   aren't there yet (the first run downloads several GB):

   ```bash
   python -m pipeline.feature_table.build_idsse_features
   python -m pipeline.feature_table.build_idsse_features --degraded
   ```

3. Train and tune the models (~5 min). This writes `models/xgb_*.json` and `models/settings.json`:

   ```bash
   python -m pipeline.model.train_six_tune_seventh --tune --refit
   ```

4. Tune the smoothing (~1 min). This writes `models/smoothing.json`:

   ```bash
   python -m pipeline.model.smoothing
   ```

5. Label our match and score it:

   ```bash
   python -m pipeline.model.predict_our_match
   python -m pipeline.model.evaluate_our_match
   ```

   `predict_our_match` prints the official scoreboard. `evaluate_our_match` breaks the same score
   down into dead vs live and H vs A.

## What we did

**0. Loading the data** ([data.py](data.py)). One loader for the IDSSE feature tables: 553k labelled
frames from 7 matches, 68 features each, label 1 = home, 0 = away. Missing values stay as NaN,
because XGBoost handles them. We always split by match, never by frame: neighbouring frames are
near-copies, so a random split leaks answers.

**1. Clean vs degraded, small test** ([compare_clean_degraded.py](compare_clean_degraded.py)).
IDSSE almost always sees the ball. Our tracking loses it 27 % of the time. The degraded table
blanks IDSSE to look like ours. We trained on 2 matches and tested on a 3rd (three rotations).
Result: no clear winner. Two matches were too few to learn from the gaps.

**2. Train on 6, tune on the 7th** ([train_six_tune_seventh.py](train_six_tune_seventh.py)).
We tried three training sets (clean, degraded, both stacked) and picked tree count and depth on
the held-out match, always scoring its *degraded* version. With six matches, degraded data
clearly helps: +2 points accuracy, +7–11 when the ball is missing. "Both" was best on all three
validation matches we tried. Final model: both, depth 8, retrained on all 7 matches.

**3. Our five minutes** ([predict_our_match.py](predict_our_match.py)). We build our match's
features the same way, predict P(home) on live frames, set dead balls to D, and score. Our
inputs look like degraded IDSSE (ball missing 26.6 % vs 26.7 %), and the model transfers well
frame by frame (macro-F1 0.878). But unsmoothed it flickers: 56 possessions instead of 17.

**4. Tuned smoothing** ([smoothing.py](smoothing.py)). We feed the model's confidence into the Viterbi decoder ([decoder.py](../decoder.py)). Every switch between teams costs a penalty
`p_switch`, so a brief blip is ignored and only sustained evidence flips the label. We chose
`p_switch` **on IDSSE, not on our five minutes**: each IDSSE match was predicted by a model that
never saw it, decoded with each penalty, and scored with the official scorer. `p_switch = 6`
scored best (spell-F1 0.35 → 0.61 on IDSSE). Viterbi also beat simple rolling averages.

**5. Applying the dead mask** ([evaluate_our_match.py](evaluate_our_match.py)). The model only
knows home vs away: IDSSE training frames are all live. So every label comes from two jobs.
Task 4's dead-ball mask decides dead or live, and anything dead becomes D (so does the second
half before kickoff). The model, smoothed by Viterbi, decides H or A on the live frames, and a
team change across a dead ball costs nothing. This script scores the two jobs separately with
the official scorer, so its numbers match `run_evaluation.py` exactly.

## Results on our five minutes

| Submission | macro-F1 | spell-F1 | Possessions (truth: 17) |
|---|---|---|---|
| **XGBoost + tuned Viterbi** | **0.881** | **0.594** | 18 |
| baseline_smooth125 | 0.581 | 0.424 | 16 |
| baseline_smooth50 | 0.621 | 0.326 | 26 |
| XGBoost, unsmoothed | 0.878 | 0.285 | 56 |

### After the dead mask: where the frames go

All 7,500 graded frames, dead ones included (XGBoost + tuned Viterbi + Task 4's dead mask), we get the confusion matrix:

| | pred H | pred A | pred D |
|---|---|---|---|
| **truth H** | 3129 | 225 | 100 |
| **truth A** | 206 | 2291 | 80 |
| **truth D** | 159 | 74 | 1236 |

| Job | Who does it | Score |
|---|---|---|
| Dead vs live | Task 4's mask | D precision 0.873, recall 0.841, F1 0.857 (best baseline: 0.42) |
| Home vs away | XGBoost + Viterbi | 0.926 correct where both sides say live; H and A both ≈ 0.89–0.91 precision and recall |

The 431 wrong-team frames (225 + 206) and 413 dead/live mistakes (233 dead frames called live,
180 live frames called D) are about even, so improving either job pays off equally. The wrong-team
errors are balanced too, so the model does not lean towards either team.

## Tried and not adopted

- **Change-aware Viterbi** (branch `change-aware-viterbi`; the code was removed from `main`). A second model predicts when possession is about to change, and the
  switch penalty drops at those moments. On IDSSE it scored spell-F1 0.621 against 0.613 today,
  better on 5 of 7 matches. That's below the +0.01 bar we set beforehand, so we kept the simpler
  decoder.

## Things to know

- **The five minutes are our only real test.** Every choice above was made on IDSSE. Keep it
  that way: picking settings by watching this scoreboard would overfit to 17 possessions.
- **17 possessions is a small sample.** The scorer's own noise band for spell-F1 is about ±0.04.
- **IDSSE labels are finer than ours.** IDSSE's provider-labelled possessions average 7.4 s, ours
  14.2 s. It still worked here, but it is a reason the best penalty for us could be a bit higher.
- **D comes from Task 4's dead-ball mask.** The model only decides H vs A, so a better dead mask
  improves this directly. Teaching the model dead balls would mean rebuilding the IDSSE tables
  with dead frames kept (they are dropped today), so talk to the Task 4 owner first.
- `models/` is gitignored. Rebuild it with the steps at the top.
