# McDonald's Workers — Ball Possession

*More than a Hack 2027, Phase 1 · technical document*

<!-- Figures are in submission/figures/, rebuilt by `python -m submission.make_figures` from our own
     pipeline outputs and IDSSE results only: no video frames, no positions. The two diagrams are
     rendered from figures/*.mmd by `node submission/render_diagrams.cjs` (instructions inside), in the
     same font and palette as the matplotlib figures. -->

## At a glance

| | |
|---|---|
| **The task** | For every frame of BAR-ATH, say which team has the ball (`H` / `A`) or that it is dead (`D`) |
| **Our approach** | *Who has the ball?* An XGBoost model trained on seven public matches. *Is the ball alive?* Rules we wrote, each tied to something visible in the video. A Viterbi decoder joins the frame-by-frame answers into possessions |
| **What we added** | our own ball detector on the tactical video, the referee's whistle from the audio, replays from the broadcast |
| **Public five minutes** | **spellF1 0.671, macroF1 0.948** (best published baseline: 0.326, 0.621). The ground truth is used only for checking, never by the pipeline |

**Key terms.**
**IDSSE**: a [public dataset](https://github.com/spoho-datascience/idsse-data) of tracking and possession labels from seven Bundesliga matches (CC-BY-4.0). You don't need it to reproduce our file, because the trained model is in `weights/`. To retrain, run `python -m pipeline.feature_table.build_idsse_features`, which downloads it into `pipeline/cache/idsse/`.
**Log-odds**: the model's probability that Barcelona has the ball on an unbounded scale (positive leans Barcelona), so evidence from different sources can be added.
**Switch cost**: what the Viterbi decoder charges each time the label changes team, so a one-second blip can't flip possession but sustained evidence can.
**The viewer**: a local web page we built (`python -m pipeline.viewer`) that plays the video next to our labels and every signal behind them.

![Public window: ground truth, our labels and the baseline](figures/public_window_strips.png)

*Figure 1. The public five minutes. We get the long possessions and all four stoppages right and miss a
few short ones; the baseline flickers.*

## 1. How we thought about the problem

- **The ball is missing a lot.** The tracking loses it on 41% of graded frames, mostly in duels,
  crowded penalty areas and fast counter-attacks.
- **The metric rewards whole possessions.** One wrong frame can split a long possession in two, so a
  model right on 91% of frames still scores badly if it flickers.
- **Dead balls are their own problem.** A missed throw-in merges two possessions; an invented stoppage
  splits one.

So we split the task. *Is the ball alive?* goes to rules that can use evidence a learned model never
sees: the ball over a line, the whistle, players stopping, the broadcast cutting to a replay. *Which
team has it?* goes to a model trained on IDSSE, which has good possession labels. *When does a
possession start and end?* goes to the decoder. 
With only five minutes of ground truth, we checked each part where it could honestly be measured:
**IDSSE out of fold** for the model and tracking-only rules; a **hidden-ball test** (hide the ball
where it *is* tracked, see if the pipeline recovers who has it) across the whole match; **the public
window** as a smoke test only (with 17 possessions, 0.04 spellF1 is noise); and **the viewer**, to
find the football reason behind each error.

## 2. The pipeline

<img src="figures/pipeline.png" alt="The pipeline" width="100%">

*Figure 2. The pipeline (source: `figures/pipeline.mmd`).*

## 3. Data we used, and what we dropped

From the provided assets we use the **tactical tracking** (features and every dead-ball rule), the
**tactical video and its homography** (our ball detector, and checking there's a ball under the
tracked one), the **video's audio** (the whistle), the **broadcast tracking** (cut-aways to replays)
and **Smart Tagging** (`SET PIECES` and `GOALS` as stoppage anchors). The only external data is IDSSE.

| tried and dropped | why |
|---|---|
| Smart Tagging as team evidence | long, coarse intervals that smothered short possessions |
| Touch and contact detection | near a crowded ball the positions are too noisy to say who touched it |
| A change-aware decoder | IDSSE spellF1 0.621 against 0.613: too small to tell from noise |
| Sequence models | no gain over XGBoost + Viterbi, and harder to reproduce |
| Requiring a "touch" for a possession | one-touch passes and tracking lag hide real touches too |

## 4. Who has the ball?

### 4.1 The model and its training data

The model sees 68 features per frame, built by the same code for our match and for IDSSE: who's near
the ball, who had it last, how the ball moves, where it is (flipped so Barcelona always attack the
same way), the shape of both teams, whether the ball is observed or missing, and key features 0.4 s
and 1 s either side.

IDSSE's tracking almost never loses the ball; ours loses it 41% of the time. So `degrade.py` blanks
IDSSE positions to match our gap statistics (26.7% of live frames, median gap 6.8 s, 2.4× likelier
to start in a duel). On a held-out match, accuracy with the ball missing rises from 0.850 to 0.895
with degraded copies in training, at no cost when the ball is tracked. The final model (depth 8)
trains on clean and degraded copies of all seven matches.

### 4.2 Fixing the tracking first

- **Goalkeepers belong to their goal.** The tracker swaps keepers between team slots: Athletic's slot
  spends 656 s in Barcelona's goal, creating 7 false changes of possession. A keeper now counts for
  the team whose goal they're in.
- **A cleaner ball track.** We remove jumps faster than 40 m/s and interpolate gaps up to 25 frames:
  90% of such fills land within 1.5 m, inside a player's ~2 m control distance. Longer gaps stay
  missing, and the model is told so.

### 4.3 From frames to possessions: the decoder

Left alone, the model reports 56 possessions in the public window, where there are 17. The decoder
finds the cheapest labelling of each half, paying the switch cost on every change of team. Dead
frames are forced to `D`, and entering or leaving `D` is free, so a restart can go to either team.

![Switch cost on IDSSE and on our match](figures/switch_cost.png)

*Figure 3. Left: IDSSE's labels favour a switch cost of 6. Right: we use 24, close to the public
possession rate. IDSSE counts every brief touch as a possession; this competition doesn't.*

We didn't stop at 16, where the curve meets the public rate: that would fit five minutes of one
match, and the opening minutes are usually the most frantic, so the full match needs a higher cost.

![The viewer's playhead plot](figures/viewer_playhead.png)

*Figure 4. The viewer around frame 12,452: a brief dip below zero can't pay the switch cost; a
sustained one switches us to Athletic, with the ground truth.*

### 4.4 Bringing in the video ball

Where the tracking has lost the ball and our detector finds it, the closer team gets a vote of up to
±2 on top of the log-odds. In the hidden-ball test (300 tracked stretches hidden), agreement with the
hidden tracking rises from 0.763 to 0.809 at weight 2.

## 5. Finding the ball in the video

[WASB](https://arxiv.org/abs/2311.05237), a published ball detector, looks at three frames at once, so it uses the ball's movement as well as its appearance in a single
frame. We adapted it to this camera with algorithm-made labels only:
re-calibrating its input
statistics on our video ([AdaBN](https://arxiv.org/abs/1603.04779)); projecting the tracked ball through the homography and snapping it
to the nearest small white blob (16,719 tiles); pasting real players over the ball on 40% of tiles,
since gaps are mostly occluded balls; and fine-tuning on held-out 10-minute stretches. It finds the
ball on 27,574 of the 64,643 graded frames the tracking lost. Training takes 11.6 h, once; the weights are committed in `weights/`.

## 6. Is the ball alive?

A frame is dead from the moment the ball crosses a line or the referee stops play until the restart
kick.

<img src="figures/dead_ball.png" alt="The dead-ball mask" width="70%">

*Figure 5. The dead-ball mask (source: `figures/dead_ball.mmd`).*

**Stage 1: six indicators vote.** Weights come from each indicator's precision, not fitting; a frame
is dead below −0.6, so one strong or two weaker indicators must agree. The vote is precise (0.95 on the public window) but misses short stoppages (recall 0.82),
and its edges are loose.
<!-- TODO (2): say where the indicator weights are in the repo (WEIGHTS and DEAD_THRESHOLD in pipeline/dead_ball.py) -->

**Stage 2: fifteen refinement steps**, in a fixed order:
- **Add missed stoppages** at set-piece tags, at broadcast cut-aways (only when something else agrees,
  as they also fire while a keeper holds the ball) and at the whistle, found by the referee's two
  steady tones (about 3.8 and 4.3 kHz) against the crowd's single gliding one.
- **Put edges on real events:** the ball leaving the pitch, the restart kick, half a second after a
  whistle; and hold the stoppage while the ball is just rolled into place.
- **Tidy up:** join stoppages split by a held ball, and drop those under 2 s (0.8% of IDSSE's).

Tracking-only steps were validated on IDSSE: edge snapping and ball-over-a-line lift spellF1 from
0.436 to 0.499, set-piece anchors to 0.522, better on all seven matches. The rest came from watching
where the mask went wrong. Cut-aways give the biggest public gain; final dead-ball F1 is 0.966.

## 7. Our possession rules

A change needs the other team to gain *controlled* possession, so loose balls, touches and aerial
duels can't pay the switch cost on their own; restarts go to whoever has the ball next.

Watching our labels in the viewer, we saw that many changes of possession were invented: the model
gave the ball to a team that, on the video, never had it. Two football principles fix most of them,
and both are deliberately strict, firing only when the ball is clearly absent:
- **No possession without the ball.** A short spell inside the other team's possession, during which
  neither the tracking nor our detector ever saw the ball, goes back to that team.
- **No possession without a ball under the tracking.** Sometimes the tracked ball sits where the
  video shows none, and the model follows it to the nearest player. A short spell in which almost no
  frame shows a ball there goes back to the team before.

## 8. Where it still goes wrong

- **Passes that go past an opponent** are our most common mistake: the ball really is next to that
  player, so every signal agrees with the model. Telling control from a ball going past needs more
  than position: we would need to follow the ball's path at the player, since a controlled touch
  changes its speed or direction and a ball running past doesn't.
- **Short possessions inside the other team's spell** (0.8–2.8 s) are our costliest public errors,
  absorbed by the switch cost.
- **Stoppage edges when the ball is lost** fall back on player movement, up to 1.5 s off.
- **Match-specific rules:** whistle tones and broadcast offset need re-measuring.

## 9. Reproducing the submission

The trained weights are committed in [`weights/`](../weights/), so nothing needs retraining: the
possession model (`xgb_both_all7.json`) and the fine-tuned ball detector (`ball_video_finetuned.pt`), and
the detector's output on our video (`ball_video.npz`) is not: see [The video ball](#the-video-ball-ball_videonpz) below.

| step | command | time |
|---|---|---|
| **Requirements** | **Python 3.12** exactly (newer versions such as 3.15 can't install the pinned scipy and torch); macOS or Linux (on macOS also `brew install libomp`); the organisers' match data in `./data`, or wherever `MTAH_DATA_DIR` points (section 0 lists the files) | — |
| **Pipeline environment** | `python3.12 -m venv --clear .venv && .venv/bin/pip install -r pipeline/requirements.txt` | minutes |
| **Video ball** | build it or email us for it: [The video ball](#the-video-ball-ball_videonpz) below | ~11.6 h, or none |
| **Submission** | `.venv/bin/python -m pipeline.submission --name mcdonalds_workers` | ~3 min fresh, under 1 min cached |
| **Check** (optional, the run already validates) | `.venv/bin/python evaluation/validate_submission.py evaluation/results/mcdonalds_workers.csv` | seconds |

The run prints the submission file's sha256, identical on every run given the same `ball_video.npz`.

### The video ball (`ball_video.npz`)

Our ball detector's output on the match video (the ball's position on the 27,574 graded frames the
tracking lost) is **not in this repository**: it is frame-level data derived from the match video, which
the data terms don't allow us to publish. The submission needs it, so either:

- **Build it** with the two commands below (~11.6 h on an Apple M1; resumes if interrupted). This only
  runs the committed fine-tuned detector (`weights/ball_video_finetuned.pt`) over the video; nothing is
  trained.
- **Or email us for our copy:** chi.tsang.25@ucl.ac.uk or sui.lim.25@ucl.ac.uk. Put it at
  `weights/ball_video.npz` (or `pipeline/cache/ball_video/ball_video.npz`).

| step | command | time |
|---|---|---|
| **Video environment** | `python3.12 -m venv --clear .venv-video && .venv-video/bin/pip install -r requirements-video.txt` | minutes |
| **Video ball** | `.venv-video/bin/python -c "from pipeline import ball_video; ball_video.build()"` | ~11.6 h on an Apple M1; resumes if interrupted |

Our copy has sha256 `fec098b93c719fb5f4b8e95fad013a5c57e925315db37d1df9d2e764ff5e2689`
(`shasum -a 256 pipeline/cache/ball_video/ball_video.npz`).

**Getting exactly our result.** The build has no random step, so there is no seed to set: runs on
the same machine and versions should give the same file. Ours was built on an Apple M1 with PyTorch's MPS backend and the
versions pinned in `requirements-video.txt` (torch 2.9.1, torchvision 0.24.1, opencv-python-headless
5.0.0.93, which also decodes the video). On other hardware (CPU or CUDA) floating-point differences can
tip a borderline detection either way, so the file, and then a few labels in the submission, can
differ. If your sha256 doesn't match ours, email us for our copy.

**Retraining the detector** is not needed (its weights are committed) and is not bit-for-bit
reproducible: training on a GPU or MPS isn't deterministic. Every random step uses **seed 0**, the
default of `ball_video.make_adabn`, `occlusion.build_donor_bank`, `ball_video_data.build` and
`ball_video_train.train` (which also sets `torch.manual_seed(0)` and seeds the data loader), so a retrain
comes close to the committed weights but not exactly. The possession model's XGBoost training samples
nothing at random and gives the same model every time.
