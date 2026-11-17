"""The submission: XGBoost possession model + video ball, Viterbi, refined dead-ball mask.

    python -m pipeline.submission                    # evaluation/results/McDonald's_Workers.csv, validated
    python -m pipeline.submission --name <team>      # name the file after the team before submitting

Stages, all deterministic:
1. P(home) on every graded frame from the XGBoost model trained on 7 IDSSE matches
   (pipeline/model/, committed as weights/xgb_both_all7.json), on our match's feature table. Goalkeeper tracks
   count for the team whose goal they stand in (pipeline/keepers.py): the tracker puts either
   keeper into either team's keeper slot. Public spellF1 0.624 -> 0.650.
2. Video ball inside tracking gaps (pipeline/ball_video.py) added to the model's log-odds, weight 2:
   hidden-ball test 0.780 -> 0.809.
3. Dead ball: Task 4's stage (pipeline/dead_ball.py), refined by pipeline/dead_refine.py (anchors,
   ball seen over a line, snapped edges, carried-ball starts trimmed, whistled starts timed by the
   referee's whistle in the video's audio (pipeline/whistle.py), stoppages under 2 s dropped;
   IDSSE spellF1 0.436 -> 0.524), plus the second half's
   pre-kickoff, which ends exactly at the known kickoff. Anchors: Smart Tagging set pieces (not the
   second half's kickoff tag), and the cuts of the TV broadcast away from play that something else
   backs up (pipeline/broadcast.py; public dead-ball F1 0.893 -> 0.932). An anchor's stoppage starts
   after the video ball's last sustained movement where the tracking lost a ball still in play.
4. Viterbi (pipeline/decoder.py), switch cost 24 (P_SWITCH).

The trained weights are committed in weights/ (the possession model and the fine-tuned ball
detector), so nothing needs training. The detector's output on our video (ball_video.npz) is not
committed: build it as below, or email chi.tsang.25@ucl.ac.uk or sui.lim.25@ucl.ac.uk for our copy
and put it at weights/ball_video.npz. To retrain the model instead (see pipeline/model/README.md),
    python -m pipeline.model.train_six_tune_seventh --tune --refit
and copy pipeline/model/models/xgb_both_all7.json to weights/. To build the video detections, with the video environment (requirements-video.txt, ~11.6 h on MPS):
    python -c "from pipeline import ball_video; ball_video.build()"
XGBoost on macOS needs the OpenMP runtime: brew install libomp. The whistle needs PyAV once
(pip install av) to read the video's audio; the result is cached.
"""

import argparse
import hashlib
import os
import sys

if __name__ == "__main__" and not {"-h", "--help"} & set(sys.argv[1:]):
    # Python version, packages, match data and weights, checked before numpy is imported, so a wrong
    # setup gets a plain message and its fix instead of a traceback (pipeline/startup.py).
    from pipeline import startup
    startup.run()

import numpy as np

from pipeline import clues, decoder as dc, progress
from pipeline.broadcast import away_starts
from pipeline.common import (BALL_VIDEO_PATH, N_FRAMES, PITCH, SECOND_HALF_PREKICKOFF, half_of_frame, load_stage,
                             load_tracking, runs, submit)
from pipeline.dead_ball import load_tags
from pipeline.dead_refine import refine
from pipeline.keepers import keepers_by_goal
from pipeline.whistle import whistle_onsets

# Params
NAME = "McDonald's_Workers"     # our team name: the rules ask for the file to be named after it
MODEL = "xgb_both_all7"
W_VIDEO = 2.0
P_SWITCH = 24.0
MERGE_UNSEEN = True   # a possession during which the ball is never seen goes back to the team either side
MERGE_BALLLESS = True # a possession whose tracked ball has no ball under it goes back to the team before


def video_ball():
    """(N, 2) video ball in metres, NaN where the detector found nothing or the tracking has the ball.
    Read from the cache, where ball_video.build() writes it, else from weights/ (our copy, on request)."""
    from pipeline.common import WEIGHTS_DIR
    committed = os.path.join(WEIGHTS_DIR, "ball_video.npz")
    path = BALL_VIDEO_PATH if os.path.exists(BALL_VIDEO_PATH) else committed
    if not os.path.exists(path):
        raise FileNotFoundError(f"neither {BALL_VIDEO_PATH} nor {committed} exists: run ball_video.build(), "
                                "or email chi.tsang.25@ucl.ac.uk or sui.lim.25@ucl.ac.uk for our copy (see README.md)")
    with np.load(path) as f:
        return f["ball_video"]


def p_home(explain=False):
    """(N,) P(home) on every graded frame (dead ones too, so any dead mask can be applied), 0.5 elsewhere.
    With `explain`, also each frame's split of the model's log-odds into one contribution per feature
    (XGBoost's exact TreeSHAP): {"frames", "features", "values" (rows, F), "contribs" (rows, F + 1, the
    last column the base value)}; each row of contribs sums to that frame's log-odds before clipping."""
    import xgboost as xgb
    from pipeline.feature_table.build_features import build_feature_table, load_our_match
    from pipeline.model.data import feature_columns
    from pipeline.common import WEIGHTS_DIR

    path = os.path.join(WEIGHTS_DIR, f"{MODEL}.json")
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} is missing: it is committed in weights/ (see this module's docstring)")
    model = xgb.XGBClassifier()
    model.load_model(path)
    table = build_feature_table(load_our_match(), drop_dead=False)
    X = table[feature_columns()].astype(np.float32)
    p = np.full(N_FRAMES, 0.5)
    p[table["frame"].to_numpy()] = model.predict_proba(X)[:, 1]
    if not explain:
        return p
    contribs = model.get_booster().predict(xgb.DMatrix(X), pred_contribs=True)
    return p, {"frames": table["frame"].to_numpy(), "features": feature_columns(), "values": X.to_numpy(),
               "contribs": contribs}


def p_switch():
    """The decoder's H<->A switch cost (P_SWITCH: why it isn't smoothing.py's IDSSE value)."""
    return P_SWITCH


def dead_inputs(half):
    """What dead_refine works from: the filled ball, the set-piece anchors, players, whistle onsets,
    broadcast cut-aways and the video ball. The viewer's diagnostics read the same inputs."""
    tags = load_tags()
    tags = tags[tags["code"] == "SET PIECES"]
    first, last = SECOND_HALF_PREKICKOFF
    # The second half's kickoff has its own tag, inside the pre-kickoff. Its frame is known, and the
    # ball is tracked moving before it, so the tag's stoppage would run 14.6 s into play: leave it out.
    tags = tags[(tags["start_frame"] < first) | (tags["start_frame"] > last)]
    return {"ball": load_stage("ball_filled")["ball"].astype(float), "tag_starts": tags["start_frame"].to_numpy(),
            "tag_ends": tags["end_frame"].to_numpy(), "players": load_tracking()["players"].astype(float),
            "whistles": whistle_onsets(half), "cutaways": away_starts(half), "video": video_ball(),
            "team": load_tracking()["team"]}


def dead_mask(half, trace=None):
    """(N,) Task 4's dead mask refined by dead_refine, graded frames only, plus the pre-kickoff. With
    a list as `trace`, each refinement step's (name, mask) is appended, then the pre-kickoff's."""
    x = dead_inputs(half)
    first, last = SECOND_HALF_PREKICKOFF
    task4 = load_stage("dead_ball")
    dead = refine(task4["dead"], x["ball"], half, x["tag_starts"], PITCH, x["players"], x["whistles"],
                  tag_ends=x["tag_ends"], cutaways=x["cutaways"], video=x["video"], trace=trace,
                  support=task4.get("support"), team=x["team"])
    dead[first:last + 1] = True
    kickoff = last + 1                                   # known exactly, so the stoppage ends exactly here
    dead[kickoff:kickoff + int(np.argmin(dead[kickoff:]))] = False
    if trace is not None:
        trace.append(("prekickoff", dead.copy()))
    return dead


def build_components(explain=False):
    """Every stage's output, per frame: the model's log-odds, the weighted video-ball clue, the dead
    mask and its mask after each refinement step, the switch cost, and the decoded labels (what
    build_labels returns). With `explain`, also the model's per-feature contributions (p_home's).
    The viewer's discrepancy report reads these (python -m pipeline.viewer --diagnostics)."""
    from pipeline.model.smoothing import log_odds

    with progress.step("Tracking: players and ball for every frame"):
        trk = load_tracking()
        half = half_of_frame()
        progress.note(f"{N_FRAMES:,} frames, {int((half > 0).sum()):,} of them graded")
    with progress.step("Ball track: remove teleports, fill short gaps"):
        load_stage("ball_filled")
    with progress.step("Possession model: 68 features per frame, XGBoost P(home)"):
        p, explained = p_home(explain=True) if explain else (p_home(), None)
        lo = log_odds(p)
    with progress.step("Video ball: our detector's ball where the tracking lost it"):
        players, team = keepers_by_goal(trk["players"], trk["team"], half)       # as the model's features
        vb = video_ball()
        progress.note(f"ball found on {int(np.isfinite(vb[:, 0]).sum()):,} frames the tracking missed")
        video = clues.inferred_ball_clue(vb, players.astype(float), team, W_VIDEO)
        score = lo + video
    with progress.step("Dead-ball mask: six indicators vote, then the refinement steps"):
        dead_steps = []
        dead = dead_mask(half, trace=dead_steps)
        progress.note(f"dead on {dead[half > 0].mean():.1%} of graded frames")
    with progress.step("Possessions: Viterbi decoder per half, then the possession rules"):
        cost = p_switch()
        labels = np.full(N_FRAMES, "X", dtype="<U1")
        for value, start, end in runs(half):
            if value:
                seg = np.where(dead[start:end], 0.0, score[start:end])
                labels[start:end] = dc.LETTERS[dc.viterbi(dc.emissions(seg, dead[start:end]), cost)]
        if MERGE_UNSEEN:
            seen = np.isfinite(load_stage("ball_filled")["ball"][:, 0]) | np.isfinite(vb[:, 0])
            labels = dc.merge_unseen_spells(labels, seen)
        before_ballless = labels.copy()
        if MERGE_BALLLESS:
            labels = dc.merge_ballless_spells(labels, load_stage("ball_seen")["white"])
        n_poss = sum(1 for v, s, e in runs(labels) if v in "HA")
        progress.note(f"{n_poss} possessions")
    out = {"log_odds": lo, "video": video, "dead": dead, "dead_steps": dead_steps, "p_switch": cost, "labels": labels,
           "ballless_merged": before_ballless != labels}
    if explain:
        out["explain"] = explained
    return out


def build_labels():
    """(N,) H / A / D / X for every frame of the tactical video: the pipeline's own labels."""
    return build_components()["labels"]


def main(name=NAME):
    progress.begin(f"Rebuilding the submission: evaluation/results/{name}.csv")
    labels = build_labels()
    with progress.step("Write the file and run the official validator"):
        path = submit(name, labels)
        with open(path, "rb") as f:
            print(f"{path}\nsha256 {hashlib.sha256(f.read()).hexdigest()}")
    progress.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--name", default=NAME, help="file name without .csv: the team name on the board")
    parser.add_argument("--no-end-at-go", action="store_true",
                        help="don't end untracked stoppages where the players get going (dead_refine.END_AT_GO)")
    parser.add_argument("--no-snap-guard", action="store_true",
                        help="let a stoppage's start snap back across a restart kick (dead_refine.SNAP_GUARD)")
    args = parser.parse_args()
    from pipeline import dead_refine as _dr
    _dr.END_AT_GO = not args.no_end_at_go
    _dr.SNAP_GUARD = not args.no_snap_guard
    main(args.name)
