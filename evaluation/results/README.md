# Results on the public five minutes

The organisers publish ground truth for one five-minute window of BAR-ATH: video frames
11,185–18,684, 7,500 frames at 25 fps. In that window 17 possessions and 20 transitions are
labelled, and the frames split into 3,454 Barcelona (`H`), 2,577 Athletic (`A`) and 1,469 dead (`D`).
Every other frame of the match is graded privately. The pipeline never reads this ground truth: we
used it only to check our labels.

These are the scores of our submitted file, `mcdonalds_workers.csv` (sha256 `306bb265…a834ead4`),
next to the four baselines committed in this folder. The baselines label each frame with the team of
the player nearest the ball, then take a majority vote over a window of ±0, ±25, ±50 or ±125 frames.
`baseline_smooth50` is the published baseline.

| submission | spellF1 | macroF1 | possessions (gt 17) | mean possession | transition F1 (±25 frames) |
|---|---|---|---|---|---|
| **mcdonalds_workers** | **0.671** | **0.948** | 11 | 21.7 s | **0.647** |
| baseline_smooth125 | 0.424 | 0.581 | 16 | 18.5 s | 0.167 |
| baseline_smooth50 | 0.326 | 0.621 | 26 | 10.8 s | 0.082 |
| baseline_smooth25 | 0.287 | 0.620 | 36 | 7.6 s | 0.203 |
| baseline_raw | 0.038 | 0.597 | 131 | 2.0 s | 0.141 |

The true mean possession in the window is 14.2 s.

## What the numbers say

- **spellF1** scores whole possessions: a predicted possession counts only if it overlaps the true
  one well enough. **macroF1** scores single frames. The baselines get most frames right, but split
  each possession into many short ones (131 for `baseline_raw` against 17 true ones), which scores
  well on macroF1 and badly on spellF1. Our decoder charges a cost for each switch, so blips can't
  flip possession: at the IoU 0.5 threshold, 10 of our 11 possessions match a true one.
- **Dead ball is almost solved here.** Alive/dead F1 is 0.991 and the `D` class F1 is 0.966: all four
  stoppages in the window are found, against a `D` F1 of 0.08–0.42 for the baselines.
- **Our main error is merging possessions.** We predict 11 possessions where there are 17, and our
  mean possession is 53% too long: the switch cost that removes blips also absorbs some real, short
  possessions. Of the frames given to the wrong team, 231 Barcelona frames are labelled Athletic and
  84 the other way round.
- **Team accuracy while the ball is in play** is 0.935, and our Barcelona share of possession is
  54.5% against the true 57.3%.

### Our file in detail

| measure | value |
|---|---|
| macroF1 (H / A / D) | 0.948 (0.943 / 0.934 / 0.966) |
| Cohen's kappa · frame accuracy | 0.912 · 0.944 |
| spellF1 at IoU 0.3 / 0.5 / 0.7 | 0.786 / 0.714 / 0.500 |
| possessions at IoU 0.5 | 10 matched, 7 missed, 1 spurious |
| transitions within ±12 / ±25 frames | F1 0.529 / 0.647 (14 predicted, 20 true) |
| transition timing | median +1 frame late, 90% within 20 frames |

### How much to trust them

The window holds only 17 possessions, so one possession more or less moves spellF1 by several
hundredths: differences under about 0.04 are noise. We used it to check the pipeline was right, not
to choose between versions. These scores cover only the public window. The final grade comes from the organisers' hidden
ground truth for the rest of the match (about 159,000 graded frames in total).

## The files

The four `baseline_*.csv` files are committed because they are built from the tracking alone and
carry no information about the answer. Our own file is not committed: a full set of labels for the
match is close to a copy of the ground truth, so it stays out of a public repository. The top-level
README explains how to rebuild it.
