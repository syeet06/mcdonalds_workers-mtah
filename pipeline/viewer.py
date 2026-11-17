"""Possession viewer: play the tactical video with our predictions, the public ground truth and the
ball drawn on it. A local server (127.0.0.1 only) feeds pipeline/viewer.html.

    python3 -m pipeline.viewer            # opens http://127.0.0.1:8765
    python3 -m pipeline.viewer --rebuild  # re-export ball pixels after a new ball_video build

The page keeps the tracks you added in the browser and, on every load, re-reads the ones that live
in evaluation/results/, so a rebuilt CSV shows up on a reload (CSVs dropped onto the page stay as
saved). http://127.0.0.1:8765/?tracks=a.csv,b.csv starts from exactly those files instead (and
?frame=12345 opens at a frame). The match clock counts each half from its kickoff (common.KICKOFFS).

Sound is off until you tick it (or press m): the stadium audio, whistles included.

The discrepancy report (side panel) compares any two tracks. Its numbers (log-odds, video clue, dead
mask, Viterbi decision) come from the current pipeline, cached once with
    python -m pipeline.viewer --diagnostics   # runs pipeline.submission's stages; video environment
    python -m pipeline.viewer --diagnostics-only   # the same, then exit (a running viewer: reload its tab)
Notes typed on the report's boxes are saved as you type to docs/report_notes.csv (one row per
submission, reference and discrepancy). It is gitignored: to share yours, send the file to a teammate, who saves it as
docs/report_notes_<your name>.csv. Every docs/report_notes_*.csv is shown read-only under the boxes.

Submissions in evaluation/results/ are listed in the page; any other CSV can be dropped onto it.
Nothing leaves the machine and nothing derived from data/ is written outside pipeline/cache/.
"""

import argparse
import csv
import errno
import http.server
import json
import os
import re
import sys
import threading
import webbrowser
from datetime import datetime, timezone

import numpy as np

from pipeline.common import CACHE_DIR, FPS, HALVES, KICKOFFS, N_FRAMES, PITCH, PUBLIC_GT, RESULTS_DIR

# Params
PORT = 8765
MISSING = -32768                    # int16 marker for "no ball on this frame"
BALL_PATH = os.path.join(CACHE_DIR, "viewer", "ball_px.bin")
DIAG_PATH = os.path.join(CACHE_DIR, "viewer", "diagnostics.bin")
DIAG_META_PATH = os.path.join(CACHE_DIR, "viewer", "diagnostics.json")
SHAP_PATH = os.path.join(CACHE_DIR, "viewer", "shap.bin")
SHAP_FULL_PATH = os.path.join(CACHE_DIR, "viewer", "shap_contribs.bin")   # every feature, for the report's contrast
SHAP_VALUES_PATH = os.path.join(CACHE_DIR, "viewer", "shap_values.bin")
SHAP_K = 8                          # features kept per frame, the ones that moved its log-odds most
LABEL_CODES = {"X": 0, "H": 1, "A": 2, "D": 3}    # the page's codes
NOTES_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "docs", "report_notes.csv")
SHARED_NOTES_RE = re.compile(r"^report_notes_([A-Za-z0-9_.-]{1,40})\.csv$")   # teammates' files, read-only
NOTE_FIELDS = ["submission", "reference", "start_frame", "end_frame", "type", "note", "updated"]
NOTE_KEY = ("submission", "reference", "start_frame", "end_frame")
MAX_NOTE = 4000
MAX_NOTE_BODY = 64 << 10
_notes_lock = threading.Lock()
PAGE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "viewer.html")
TEAMS = {"H": "Barcelona", "A": "Athletic Club"}
CHUNK = 1 << 20


def ball_pixels(ball, H, pitch=PITCH):
    """(N, 2) int16 image pixels of pitch-metre points `ball` through per-frame homographies `H`
    (which map normalised 0-1 pitch coordinates to pixels); MISSING where either is NaN."""
    x, y = ball[:, 0] / pitch[0], ball[:, 1] / pitch[1]
    den = H[:, 2, 0] * x + H[:, 2, 1] * y + H[:, 2, 2]
    u = (H[:, 0, 0] * x + H[:, 0, 1] * y + H[:, 0, 2]) / den
    v = (H[:, 1, 0] * x + H[:, 1, 1] * y + H[:, 1, 2]) / den
    px = np.stack([u, v], axis=1)
    ok = np.isfinite(px).all(axis=1) & (np.abs(px) < 30000).all(axis=1)
    out = np.full(px.shape, MISSING, dtype=np.int16)
    out[ok] = np.round(px[ok]).astype(np.int16)
    return out


def build_ball(path=BALL_PATH):
    """Write (N, 4) int16 little-endian: tracking ball (x, y), video detection (x, y), in pixels. The
    video detection is the one the pipeline uses (ball_video.build()'s output, BALL_VIDEO_PATH), not the
    detector's resumable working file, which is partial until its ~11 h run finishes."""
    from pipeline.common import BALL_VIDEO_PATH
    from pipeline.common import load_tracking
    from starter.load_homography import load

    H, _, _ = load()
    out = np.full((N_FRAMES, 4), MISSING, dtype=np.int16)
    out[:, :2] = ball_pixels(load_tracking()["ball"], H)
    if os.path.exists(BALL_VIDEO_PATH):
        with np.load(BALL_VIDEO_PATH) as c:
            out[:, 2:] = ball_pixels(c["ball_video"], H)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    out.astype("<i2").tofile(path)
    print(f"ball pixels: tracking on {np.mean(out[:, 0] != MISSING):.0%} of frames, "
          f"video detections on {np.sum(out[:, 2] != MISSING):,} frames -> {path}")
    return path


def ball_stale(path=BALL_PATH):
    """True if the ball boxes are missing or older than what they're drawn from: the video detections
    (blue box, BALL_VIDEO_PATH) or the tracking. One built before the detections existed has no blue box."""
    from pipeline.common import BALL_VIDEO_PATH
    if not os.path.exists(path):
        return True
    sources = [BALL_VIDEO_PATH, os.path.join(CACHE_DIR, "tracking.npz")]
    return any(os.path.exists(s) and os.path.getmtime(s) > os.path.getmtime(path) for s in sources)


def read_notes(path=NOTES_PATH):
    """[{field: str}] rows of the shared notes file, [] if there is none yet."""
    if not os.path.exists(path):
        return []
    with open(path, newline="", encoding="utf-8") as f:
        return [{k: row.get(k) or "" for k in NOTE_FIELDS} for row in csv.DictReader(f)]


def shared_notes(folder=None):
    """[{field: str, "from": name}] rows of every teammate's docs/report_notes_<name>.csv."""
    folder = folder or os.path.dirname(NOTES_PATH)
    out = []
    for name in sorted(os.listdir(folder)) if os.path.isdir(folder) else []:
        m = SHARED_NOTES_RE.match(name)
        if m:
            try:
                out += [dict(r, **{"from": m.group(1)}) for r in read_notes(os.path.join(folder, name)) if r["note"]]
            except (OSError, csv.Error, UnicodeDecodeError) as e:
                print(f"skipping {name}: {e}", file=sys.stderr)
    return out


def check_note(entry):
    """A note from the page, validated and normalised to the file's fields (strings)."""
    if not isinstance(entry, dict):
        raise ValueError("note must be an object")
    sub, ref = str(entry["submission"])[:200], str(entry["reference"])[:200]
    start, end = int(entry["start_frame"]), int(entry["end_frame"])
    if not (sub and ref and 0 <= start <= end < N_FRAMES):
        raise ValueError("bad note key")
    note = str(entry.get("note") or "")[:MAX_NOTE].replace("\r\n", "\n").replace("\r", "\n").strip()
    return {"submission": sub, "reference": ref, "start_frame": str(start), "end_frame": str(end),
            "type": str(entry.get("type") or "")[:60], "note": note}


def save_note(entry, path=NOTES_PATH):
    """Add, change or (empty note) remove one note; rewrite the file atomically, sorted so git diffs
    stay small. Returns the stored row, or None if it was removed."""
    row = check_note(entry)
    key = tuple(row[k] for k in NOTE_KEY)
    with _notes_lock:
        rows = [r for r in read_notes(path) if tuple(r[k] for k in NOTE_KEY) != key]
        if row["note"]:
            row["updated"] = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
            rows.append(row)
        rows.sort(key=lambda r: (r["reference"], r["submission"], int(r["start_frame"]), int(r["end_frame"])))
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp = path + ".tmp"
        with open(tmp, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=NOTE_FIELDS)
            w.writeheader()
            w.writerows(rows)
        os.replace(tmp, path)
    return row if row["note"] else None


def refine_step_rules():
    """{step: what it does, with dead_refine's current numbers}, in the order refine runs them."""
    from pipeline import dead_refine as dr
    return {
        "trim_played_starts": f"start rests on player_slowdown alone: moves after the last played ball (>= {dr.MOVING_MS:g} m/s "
                              f"for {dr.PLAY_SUSTAIN} frames inside the pitch, reaching a player < {dr.CARRY_NEAR_M:g} m for "
                              f"{dr.CONTROL_FRAMES} frames), at most {dr.PLAY_SEARCH_S} s in, never past a whistle",
        "anchor_set_pieces": f"SET PIECES tag: dead from the ball's last movement (>= {dr.MOVING_MS:g} m/s, up to "
                             f"{dr.MAX_BACK_S} s back; video ball moving {dr.VIDEO_MOVING_FRAMES} frames on end counts) "
                             f"to the restart (within {dr.MAX_AHEAD_S} s, not past the tag's end)",
        "anchor_cutaways": f"broadcast cut-away, built like a tag's; kept only if Task 4, the ball over a line or a "
                           f"whistle agrees, from {dr.CUT_NEAR_S} s before",
        "outside_dead": f"ball > {dr.OUT_MARGIN_M:g} m over a line: dead until the restart kick (within "
                        f"{dr.MAX_WAIT_S} s) or until back inside; back inside already >= {dr.MOVING_MS:g} m/s ends it",
        "snap_dead_edges": f"edges within {dr.SNAP_WINDOW / FPS:g} s snap: start to the first frame the ball is over "
                           f"a line, end to the restart kick (leaves <= {dr.STILL_MS:g} m/s, reaches "
                           f"{dr.KICK_MS:g} m/s within {dr.RISE_FRAMES} frames)",
        "trim_carried_starts": f"first 3 s >= {dr.CARRY_SHARE:.0%} carried (player < {dr.CARRY_NEAR_M:g} m, ball "
                               f"{dr.CARRY_MS[0]:g}-{dr.CARRY_MS[1]:g} m/s): starts after {dr.CARRY_CLEAR_S:g} s with no carry",
        "start_at_whistles": f"ball not over a line near the start: starts {dr.WHISTLE_LAG} frames after a whistle "
                             f"onset from {dr.WHISTLE_BACK_S} s before it to {dr.WHISTLE_NEAR_S} s after",
        "stoppages_at_whistles": f"whistle with no stoppage within {dr.WHISTLE_NEAR_S} s: dead from "
                                 f"{dr.WHISTLE_LAG} frames after it to the restart (within {dr.MAX_WAIT_S} s)",
        "hold_through_quiet": f"ends while players are quiet (<= {dr.QUIET_PLAYERS} at >= {dr.RUN_MS:g} m/s over "
                              f"{dr.QUIET_S} s): moved to the first restart in the {dr.QUIET_S} s before they get going (within {dr.MAX_WAIT_S} s)",
        "end_at_players_go": "ball missing at the stoppage's end: it ends where the players get going "
                             f"(after their last quiet moment: <= {dr.QUIET_PLAYERS} at >= {dr.RUN_MS:g} m/s over {dr.QUIET_S} s); "
                             f"not when it already ends on a corner kick the tracking sees (within {dr.CORNER_KICK_M:g} m of a flag)",
        "bridge_still_gaps": f"live gap of at most {dr.MAX_STILL_GAP_S} s between two stoppages, ball never "
                             f">= {dr.MOVING_MS:g} m/s: dead",
        "bridge_short_gaps": f"live gap of at most {dr.MAX_SHORT_GAP_S:g} s between two stoppages: dead",
        "start_at_duel_stops": f"start too early while play goes on: moves later, at most {dr.DUEL_SEARCH_S} s, to a duel stop "
                               f"(home + away within {dr.DUEL_M:g} m, pair >= {dr.DUEL_FAST_MS:g} m/s then <= {dr.DUEL_SLOW_MS:g} m/s) "
                               f"where everyone slows >= {dr.DUEL_DROP_MS:g} m/s, players ran >= {dr.DUEL_PLAY_MS:g} m/s before it "
                               f"and the ball is seen <= {dr.DUEL_BALL_LOST:.0%} of the 2 s after",
        "stoppages_at_seen_restarts": f"ball reappears at kick speed within {dr.SEEN_RESTART_EDGE_M:g} m inside a line (or outside) "
                                      f"after >= {dr.SEEN_RESTART_GAP_S} s unseen, in play: dead from the gap's start (at most "
                                      f"{dr.SEEN_RESTART_BACK_S} s back) to that kick",
        "drop_short": f"stoppages under {dr.MIN_DEAD_S} s dropped",
        "prekickoff": "second half before its kickoff: always dead, ending exactly at the kickoff",
    }


def feature_notes(features):
    """{feature: "unit · what it is"} for the model's features, from the feature modules' own tables;
    a _prev<k> / _next<k> copy says how far it is shifted."""
    from pipeline.feature_table import features_A_B, features_C_D_F
    notes = {}
    for doc in (features_A_B.__doc__, features_C_D_F.__doc__):
        for line in (doc or "").splitlines():
            m = re.match(r"^([a-z_0-9]+) {2,}(.+?) {2,}(.+)$", line)
            if m:
                notes[m.group(1)] = f"{m.group(2)} · {m.group(3)}"
    toward = "+ towards the away goal, 0 = halfway"
    notes.update({
        "ball_x": f"metres · ball along the pitch, {toward}", "ball_y": "metres · ball across the pitch, 0 = middle",
        "ball_y_abs": "metres · ball's distance from the middle, across the pitch",
        "centroid_x_diff": "metres · home's outfield centroid x minus away's", "spread_diff": "metres · home's spread minus away's",
        "n_ahead_diff": "count · home players goal-side of the ball minus away's"})
    for t in ("home", "away"):
        notes.update({f"centroid_x_{t}": f"metres · {t} outfield players' mean x, {toward}",
                      f"centroid_vx_{t}": f"m/s · {t} outfield players' mean forward speed",
                      f"spread_{t}": f"metres · how spread out {t}'s outfield players are",
                      f"n_tracked_{t}": f"count · {t} players tracked",
                      f"n_{t}_ahead_of_ball": f"count · {t} players between the ball and their own goal"})
    out = {}
    for name in features:
        m = re.match(r"^(.+)_(prev|next)(\d+)$", name)
        if m:
            k = int(m.group(3))
            out[name] = f"{notes.get(m.group(1), m.group(1))} ({k} frames, {k / FPS:g} s {'earlier' if m.group(2) == 'prev' else 'later'})"
        else:
            out[name] = notes.get(name, "")
    return out


def write_shap(explained, path=SHAP_PATH, k=SHAP_K):
    """Write (N, 2 + 3k + 4) float32 little-endian per frame: the base value, the other features' sum,
    the k features that moved the log-odds most (index, contribution, value), biggest first, then the
    feature whose contribution changed most since the previous frame (index, change, contribution,
    value; NaN on a half's first frame). NaN where the model doesn't run."""
    contribs, values = explained["contribs"], explained["values"]
    feat = contribs[:, :-1]
    top = np.argsort(-np.abs(feat), axis=1, kind="stable")[:, :k]
    rows = np.arange(len(feat))[:, None]
    tc = feat[rows, top]
    out = np.full((N_FRAMES, 2 + 3 * k + 4), np.nan, dtype="<f4")
    f = explained["frames"]
    out[f, 0] = contribs[:, -1]
    out[f, 1] = feat.sum(axis=1) - tc.sum(axis=1)
    out[f, 2:2 + k], out[f, 2 + k:2 + 2 * k], out[f, 2 + 2 * k:2 + 3 * k] = top, tc, values[rows, top]
    # Biggest change from the previous frame, where that frame is explained too, in the same half.
    from pipeline.common import half_of_frame
    half = half_of_frame()[f]
    prev = np.flatnonzero((np.diff(f) == 1) & (half[1:] == half[:-1])) + 1
    delta = feat[prev] - feat[prev - 1]
    j = np.argmax(np.abs(delta), axis=1)
    i = np.arange(len(prev))
    c = 2 + 3 * k
    out[f[prev], c], out[f[prev], c + 1] = j, delta[i, j]
    out[f[prev], c + 2], out[f[prev], c + 3] = feat[prev, j], values[prev, j]
    out.tofile(path)
    # Every feature's contribution and value, (N, F) float16, NaN where the model doesn't run: the
    # report's "why these errors" averages them over many frames, which the top k alone would bias.
    for full_path, a in ((SHAP_FULL_PATH, feat), (SHAP_VALUES_PATH, values)):
        full = np.full((N_FRAMES, a.shape[1]), np.nan, dtype="<f2")
        with np.errstate(over="ignore"):
            full[f] = a
        full.tofile(full_path)
    return {"k": k, "width": out.shape[1], "full": True, "features": list(explained["features"]), "notes": feature_notes(explained["features"])}


def build_diagnostics(path=DIAG_PATH, meta_path=DIAG_META_PATH):
    """Write (N, C) float32 little-endian per frame, columns named in the meta file: the model's
    log-odds, the weighted video-ball clue, the final dead mask (0/1), the decoded label (page codes),
    Task 4's dead mask before dead_refine, each Task 4 indicator's vote (-1 dead, +1 live, 0 none),
    the mask after each refinement step (bit i = step i), and what those steps look at: the ball's
    speed, how far over a line it is, restart kicks, carried balls and the video ball's movement."""
    from pipeline import dead_ball as db
    from pipeline import dead_refine as dr
    from pipeline import decoder as dc
    from pipeline import submission as sub
    from pipeline.common import SECOND_HALF_PREKICKOFF, half_of_frame, load_stage

    c = sub.build_components(explain=True)
    votes = db.indicators()
    steps = c["dead_steps"]
    x = sub.dead_inputs(half_of_frame())
    ball = x["ball"]
    speed = dr.ball_speed(ball)
    with np.errstate(invalid="ignore"):
        over = np.max(np.stack([-ball[:, 0], ball[:, 0] - PITCH[0], -ball[:, 1], ball[:, 1] - PITCH[1]]), axis=0)
    cols = {"log_odds": c["log_odds"], "video": c["video"], "dead": c["dead"],
            "label": np.array([LABEL_CODES[x] for x in c["labels"]]),
            "task4_dead": load_stage("dead_ball")["dead"]}
    cols.update({f"vote_{name}": votes[name] for name in db.WEIGHTS})
    cols["refine_steps"] = sum(m.astype(np.int64) << i for i, (_, m) in enumerate(steps))
    cols.update({"ball_speed": speed, "ball_over_line_m": over, "restart_kick": dr._restarts(speed),
                 "carried": dr.carried(ball, x["players"]), "video_moving": dr._video_play(ball, x["video"], PITCH)[0]})
    cols["ball_white_px"] = load_stage("ball_seen")["white"]       # white pixels in the ball square (pipeline/ball_seen.py)
    cols["ballless_merged"] = c["ballless_merged"]                  # frames merge_ballless_spells gave back
    everyone = dr.everyone_speed(x["players"])
    cols.update({"players_speed": everyone, "players_slowdown": dr.slowdown(everyone),
                 "duel_stop": dr.duel_frames(x["players"], x["team"])})
    out = np.stack([np.asarray(v, dtype="<f4") for v in cols.values()], axis=1)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    out.tofile(path)
    shap = write_shap(c["explain"])
    cfg = db.EVAL_CONFIG
    meta = {"model": sub.MODEL, "w_video": sub.W_VIDEO, "p_switch": float(c["p_switch"]),
            "columns": list(cols), "dead_weights": dict(db.WEIGHTS), "dead_threshold": cfg["DEAD_THRESHOLD"],
            "max_live_gap_s": cfg["MAX_LIVE_GAP_S"], "min_dead_s": cfg["MIN_DEAD_S"],
            "prekickoff": list(SECOND_HALF_PREKICKOFF),
            "refine_steps": [name for name, _ in steps], "refine_rules": refine_step_rules(),
            "refine_numbers": {"still_ms": dr.STILL_MS, "moving_ms": dr.MOVING_MS, "kick_ms": dr.KICK_MS,
                               "out_margin_m": dr.OUT_MARGIN_M, "whistle_near_s": dr.WHISTLE_NEAR_S,
                               "whistle_lag": dr.WHISTLE_LAG, "carry_near_m": dr.CARRY_NEAR_M,
                               "carry_ms": list(dr.CARRY_MS), "min_dead_s": dr.MIN_DEAD_S,
                               "duel_m": dr.DUEL_M, "duel_fast_ms": dr.DUEL_FAST_MS, "duel_slow_ms": dr.DUEL_SLOW_MS,
                               "duel_search_s": dr.DUEL_SEARCH_S, "duel_drop_ms": dr.DUEL_DROP_MS,
                               "duel_play_ms": dr.DUEL_PLAY_MS, "duel_ball_lost": dr.DUEL_BALL_LOST},
            "events": {"set_piece_tags": [[int(s), int(e)] for s, e in zip(x["tag_starts"], x["tag_ends"])],
                       "cutaways": sorted(int(f) for f in x["cutaways"]),
                       "whistles": sorted(int(f) for f in x["whistles"])},
            "no_ball": {"white_px": dc.NO_BALL_WHITE_PX, "share": dc.NO_BALL_SHARE, "max_s": dc.NO_BALL_MAX_S},
            "shap": shap, "built": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
            "code": code_version()}
    with open(meta_path, "w") as f:
        json.dump(meta, f)
    print(f"diagnostics: {len(cols)} columns, built {meta['built']} -> {path}")
    return path


_tags_cache = []


def stoppage_tags():
    """[{code, type, team_name, team, start, end, kick}] for every Smart Tagging SET PIECES and GOALS tag:
    its type (Throw-in, Corner Kick, ...), team (H / A / None), first and last frame, and the first
    restart kick inside it (None if none), where Task 4's tags vote stops saying dead. Read once."""
    if _tags_cache:
        return _tags_cache[0]
    import xml.etree.ElementTree as ET
    from games import require
    from pipeline.common import load_stage
    from pipeline.dead_ball import TAGS_CONFIG
    from pipeline.dead_refine import _restarts, ball_speed

    restarts = _restarts(ball_speed(load_stage("ball_filled")["ball"].astype(float)))
    out = []
    for inst in ET.parse(require(["smart_tagging"])[0]).getroot().iter("instance"):
        code = inst.findtext("code")
        if code not in TAGS_CONFIG["stoppage_tags"]:
            continue
        labels = {l.findtext("group"): l.findtext("text") for l in inst.iter("label")}
        start, end = round(float(inst.findtext("start")) * FPS), round(float(inst.findtext("end")) * FPS)
        kick = np.flatnonzero(restarts[start:end + 1])
        team = labels.get("Team")
        out.append({"code": code, "type": labels.get("Type") or code.title(), "team_name": team,
                    "team": "H" if team == TEAMS["H"] else None if team in (None, "N/A") else "A",
                    "start": start, "end": end, "kick": int(start + kick[0]) if len(kick) else None})
    out.sort(key=lambda t: t["start"])
    _tags_cache.append(out)
    return out


def code_version():
    """A fingerprint of the pipeline's code (every .py under pipeline/): the diagnostics record the one
    they were built from, and the viewer rebuilds them when it changes, so after a git pull everyone's
    report shows what the pulled code does, not what their old code did."""
    import hashlib
    h = hashlib.sha256()
    here = os.path.dirname(os.path.abspath(__file__))
    for root, dirs, files in sorted(os.walk(here)):
        dirs[:] = sorted(d for d in dirs if d not in ("cache", "__pycache__", "tests"))
        for name in sorted(files):
            if name.endswith(".py"):
                path = os.path.join(root, name)
                h.update(os.path.relpath(path, here).encode())
                with open(path, "rb") as f:
                    h.update(f.read())
    return h.hexdigest()[:16]


def diag_meta(path=DIAG_META_PATH):
    """The diagnostics' settings, or None if they haven't been built."""
    if not (os.path.exists(path) and os.path.exists(DIAG_PATH)):
        return None
    with open(path) as f:
        return json.load(f)


def parse_range(header, size):
    """(first, last) byte positions of an HTTP Range header against a file of `size` bytes, or None."""
    m = re.fullmatch(r"bytes=(\d*)-(\d*)", header.strip())
    if not m or not (m.group(1) or m.group(2)):
        return None
    if m.group(1):
        first = int(m.group(1))
        last = min(int(m.group(2)), size - 1) if m.group(2) else size - 1
    else:
        first, last = max(0, size - int(m.group(2))), size - 1
    return (first, last) if first <= last and first < size else None


def result_names(folder=RESULTS_DIR):
    """Submission CSVs in `folder`, sorted."""
    return sorted(f for f in os.listdir(folder) if f.endswith(".csv")) if os.path.isdir(folder) else []


def result_path(name, folder=RESULTS_DIR):
    """Path of a listed submission, or None (so the page can't ask for anything else)."""
    return os.path.join(folder, name) if name in result_names(folder) else None


class Handler(http.server.BaseHTTPRequestHandler):
    video = None

    def do_GET(self):
        route = self.path.split("?")[0]
        if route == "/":
            return self._file(PAGE, "text/html; charset=utf-8")
        if route == "/meta":
            meta = {"n_frames": N_FRAMES, "fps": FPS, "halves": HALVES, "kickoffs": KICKOFFS, "teams": TEAMS,
                    "results": result_names(), "ball": os.path.exists(BALL_PATH), "diag": diag_meta()}
            return self._bytes(json.dumps(meta).encode(), "application/json")
        if route == "/video":
            return self._file(self.video, "video/mp4", ranged=True)
        if route == "/gt":
            return self._file(PUBLIC_GT, "text/csv")
        if route == "/ball":
            return self._file(BALL_PATH, "application/octet-stream")
        if route == "/diag":
            return self._file(DIAG_PATH, "application/octet-stream")
        if route == "/tags":
            return self._bytes(json.dumps({"tags": stoppage_tags()}).encode(), "application/json")
        if route == "/shap":
            return self._file(SHAP_PATH, "application/octet-stream")
        if route == "/shap_contribs":
            return self._file(SHAP_FULL_PATH, "application/octet-stream")
        if route == "/shap_values":
            return self._file(SHAP_VALUES_PATH, "application/octet-stream")
        if route == "/notes":
            return self._bytes(json.dumps({"notes": read_notes(), "shared": shared_notes()}).encode(), "application/json")
        if route.startswith("/results/"):
            path = result_path(route[len("/results/"):])
            if path:
                return self._file(path, "text/csv")
        self.send_error(404)

    def do_POST(self):
        if self.path.split("?")[0] != "/notes":
            return self.send_error(404)
        # Only this page may write: JSON only (another site can't send it without a preflight we
        # never answer), and from our own origin if the browser says where it comes from.
        origin = self.headers.get("Origin")
        if not (self.headers.get("Content-Type") or "").startswith("application/json") or \
                (origin and origin != f"http://{self.headers.get('Host')}"):
            return self.send_error(403)
        size = int(self.headers.get("Content-Length") or 0)
        if not 0 < size <= MAX_NOTE_BODY:
            return self.send_error(413)
        try:
            row = save_note(json.loads(self.rfile.read(size)))
        except (ValueError, KeyError, TypeError) as e:
            return self.send_error(400, str(e)[:200])
        return self._bytes(json.dumps({"note": row}).encode(), "application/json")

    def _bytes(self, body, ctype):
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _file(self, path, ctype, ranged=False):
        if not path or not os.path.isfile(path):
            return self.send_error(404)
        size = os.path.getsize(path)
        first, last = 0, size - 1
        header = self.headers.get("Range") if ranged else None
        if header:
            span = parse_range(header, size)
            if span is None:
                self.send_response(416)
                self.send_header("Content-Range", f"bytes */{size}")
                return self.end_headers()
            first, last = span
            self.send_response(206)
            self.send_header("Content-Range", f"bytes {first}-{last}/{size}")
        else:
            self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(last - first + 1))
        if ranged:
            self.send_header("Accept-Ranges", "bytes")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        try:
            with open(path, "rb") as f:
                f.seek(first)
                left = last - first + 1
                while left > 0:
                    buf = f.read(min(CHUNK, left))
                    if not buf:
                        break
                    self.wfile.write(buf)
                    left -= len(buf)
        except (BrokenPipeError, ConnectionResetError):
            pass                                                 # the browser dropped a seek; normal

    def log_message(self, *args):
        pass


def main(argv=None):
    from games import require

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--port", type=int, default=PORT)
    ap.add_argument("--rebuild", action="store_true", help="re-export the ball pixels")
    ap.add_argument("--diagnostics", action="store_true",
                    help="(re)build the report's per-frame numbers from pipeline.submission's stages, then start")
    ap.add_argument("--diagnostics-only", action="store_true",
                    help="(re)build those numbers and exit without starting the viewer")
    ap.add_argument("--no-browser", action="store_true")
    args = ap.parse_args(argv)
    if args.rebuild or ball_stale():
        build_ball()
    old = diag_meta() or {}
    if args.diagnostics_only:
        build_diagnostics()
        print("reload the viewer's tab to see the new numbers")
        return 0
    stale = old.get("code") != code_version()        # missing, or built from other pipeline code (a git pull)
    if args.diagnostics or stale or "full" not in old.get("shap", {}):
        if stale and old:
            print("the pipeline's code has changed since the report's numbers were built: rebuilding them")
        try:
            build_diagnostics()
            if stale:                                # the submission the page scores, from the same code
                from pipeline import submission
                submission.main()
        except Exception as e:                       # the viewer still works, just without the report's numbers
            print(f"diagnostics not built ({type(e).__name__}: {e}); the report will list discrepancies without the "
                  f"pipeline's numbers. Fix that and run python -m pipeline.viewer --diagnostics")
    Handler.video = require(["video"])[0]
    url = f"http://127.0.0.1:{args.port}/"
    try:
        server = http.server.ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    except OSError as e:
        if e.errno != errno.EADDRINUSE:
            raise
        # Most likely a viewer already running: it reads the page and the cache fresh on every load,
        # so reloading its tab shows anything just rebuilt.
        print(f"port {args.port} is already in use, most likely by a viewer you already started: open {url} and "
              f"reload the tab (anything just rebuilt shows up). To run a second one: --port 8770")
        if not args.no_browser:
            webbrowser.open(url)
        return 0

    print(f"possession viewer at {url} (Ctrl+C to stop)")
    if not args.no_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    sys.exit(main())
