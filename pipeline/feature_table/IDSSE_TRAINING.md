# IDSSE feature table

## Message to teammates: accessing the training data

The IDSSE training table contains **553,105 live frames from seven matches**,
with **68 features per frame**, match/frame identifiers, and a possession label
(`1` = home, `0` = away).

The generated file is `pipeline/feature_table/idsse_all_matches_features.csv.gz`.
It is **gitignored**, so `git pull` alone will not download the training data.
Once the builder code has been committed and pushed, pull it and run the
generation command below from the repository root to create your own copy.
You need the project's Python dependencies, including Kloppy, and an internet
connection. The first run downloads several gigabytes and processes all seven
matches; subsequent runs reuse local checkpoints.

If you already have the generated file, load it directly with the Python example
below. Pandas reads the compressed CSV without manually unzipping it. Run that
example from the repository root so the relative paths resolve correctly.

Use the summary's `feature_columns` list for model inputs and `label` as the
target. Split training/validation/test data by `match_id`; do not use raw
possession codes or record identifiers as model inputs.

## Generate and load the data

Generate all seven complete matches from the repository root:

```bash
python3 -m pipeline.feature_table.build_idsse_features
```

Output: `pipeline/feature_table/idsse_all_matches_features.csv.gz`.
Match-level reports and the ordered model inputs are in
`idsse_all_matches_summary.json`. Raw XML and per-match checkpoints are in
`pipeline/cache/idsse/`; reruns reuse them. Delete checkpoints to rebuild after
changing feature code. A final combined file is only published after all seven
matches finish successfully.

```python
import json
from pathlib import Path
import pandas as pd

folder = Path('pipeline/feature_table')  # run from repository root
report = json.loads((folder / 'idsse_all_matches_summary.json').read_text())
table = pd.read_csv(folder / 'idsse_all_matches_features.csv.gz')
columns = report['matches'][0]['feature_columns']
labelled = table['label'].isin([0, 1])
X = table.loc[labelled, columns]
y = table.loc[labelled, 'label'].astype(int)
groups = table.loc[labelled, 'match_id']
```

Split by `match_id`, never randomly by frame. Do not put `frame`, `half`,
`match_id`, `source_frame_id`, raw annotation codes, or `label` in X.
The 68 model inputs come from the same shared `build_feature_table()` used for
the original match. Future-frame features require offline inference.

The export keeps regulation-half live frames (`BallStatus == 1`). Temporal
features are computed before filtering. `label` maps explicit raw
`BallPossession` 1 to home (1), 2 to away (0); all other values stay unknown.
Raw codes are retained for audit, avoiding Kloppy's default-to-away mapping.
These are provider annotations, not independently verified human labels.

Present provider ball positions count as observed; missing positions remain
NaN. No extra interpolation, artificial corruption, or original-match-specific
imputation/dead-ball rules are applied. Data quality can therefore differ from
the original match even though feature columns and calculations match.

## Degraded (augmented) set

IDSSE sees the ball and every player on almost every frame; our match doesn't. The degraded set
blanks positions so IDSSE looks like our tracking, so a model can't learn to rely on a ball that
is always there:

```bash
python3 -m pipeline.feature_table.build_idsse_features --degraded
```

Output: `idsse_all_matches_features_degraded.csv.gz` and `idsse_all_matches_summary_degraded.json`.
Code: [`degrade.py`](degrade.py) (tests in `pipeline/tests/test_degrade.py`).

- Ball: long gaps (lengths drawn from ours), starting more often in duels, with a slow ball and
  near either goal, until 26.7 % of live frames have no ball, as in our match.
- Players: single dropouts, groups of 3-8, rare near-total and whole-team blackouts, tuned to
  our 21.1 tracked on average, 8 % of frames under 20.
- Only blanks, never moves positions. Rows, labels, halves and the live mask are identical to
  the clean set, so the two tables line up row for row; only feature values differ.
- Each match has a fixed seed (`degrade.seed_for(match_id)`): rebuilds give the same file.
- Not modelled: ghost slots (> 11 on a team), track identity breaks, ball glitches.

**Check before comparing.** Each summary has `degradation_stats_all` (all IDSSE matches) and
`degradation_stats_our_match` (our match, when `data/` is present). In the degraded summary they
should be close; in the clean one, IDSSE will show almost no missing ball.

**Comparing clean vs degraded training.** Use the same leave-one-match-out split for both, and
score both models on the **degraded** version of each held-out match: that is what our real match
looks like. Scoring on clean held-out data would favour the model trained on clean data.

Source: https://huggingface.co/datasets/pysport/idsse-data
License: CC BY 4.0. Cite Bassek et al., *An integrated dataset of synchronized
spatiotemporal and event data in elite soccer*, Scientific Data 12, 195 (2025),
https://doi.org/10.1038/s41597-025-04505-y.
