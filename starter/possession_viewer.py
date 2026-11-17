"""Frame-by-frame viewer: tracking on a pitch, with predicted and true labels on a timeline.

Writes one self-contained HTML file. Open it in a browser, no server needed.

    python starter/possession_viewer.py evaluation/results/my_submission.csv
    python starter/possession_viewer.py my.csv --start 40000 --end 47500 --gt none

The timeline shows three strips -- ground truth, prediction, and a red strip wherever the two
disagree on a graded frame. Click anywhere on it to jump there.

    space        play / pause
    left/right   one frame (shift: one second)
    n / p        next / previous error stretch
    click        seek on the timeline

The output embeds tracking from data/, so it goes to evaluation/reports/ by default, which is
gitignored. Do not commit it or share it: it is derived match data.
"""

import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)
from games import GAME, GAMES, require  # noqa: E402
from baseline import FIELD_SEP, GROUP_SEP, MISSING, team_of_each_track  # noqa: E402

PUBLIC_GT = os.path.join(ROOT, "evaluation", "gt", "{}_possession_gt_public.csv".format(GAME))
PUBLIC_WINDOW = (11185, 18684)
OUT_DIR = os.path.join(ROOT, "evaluation", "reports")


def read_labels(path):
    """The label column of a frame,label CSV, as an array of one-letter strings."""
    with open(path) as handle:
        handle.readline()
        return np.array([line.rstrip("\r\n").split(",")[1].strip().upper() for line in handle])


def read_window(raw_path, start, end, pitch):
    """Players and ball for video frames start..end inclusive, in metres, rounded to 0.1 m.

    Per frame: {"p": [[x, y, slot], ...] present players only, "b": [x, y] or null}.
    """
    length, width = pitch
    frames = [None] * (end - start + 1)
    with open(raw_path) as handle:
        for line in handle:
            colon = line.index(FIELD_SEP)
            frame = int(line[:colon]) - 1  # frame_id counts from 1, video frames from 0
            if frame < start:
                continue
            if frame > end:
                break
            parts = line.rstrip("\n").split(FIELD_SEP)
            players = []
            for slot, group in enumerate(parts[1].split(GROUP_SEP)):
                if group[0] == MISSING:
                    continue
                x, y = group.split(",")[:2]
                players.append([round(float(x) * length, 1), round(float(y) * width, 1), slot])
            ball = None
            if parts[2][0] != MISSING:
                x, y = parts[2].split(",")[:2]
                ball = [round(float(x) * length, 1), round(float(y) * width, 1)]
            frames[frame - start] = {"p": players, "b": ball}
    return [f or {"p": [], "b": None} for f in frames]


def build_html(payload):
    with open(os.path.join(HERE, "possession_viewer.html")) as handle:
        template = handle.read()
    return template.replace("/*__PAYLOAD__*/null", json.dumps(payload, separators=(",", ":")))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("pred", help="a frame,label submission CSV")
    parser.add_argument("--gt", default=PUBLIC_GT, help="ground-truth CSV, or 'none'")
    parser.add_argument("--start", type=int, default=PUBLIC_WINDOW[0], help="first video frame")
    parser.add_argument("--end", type=int, default=PUBLIC_WINDOW[1], help="last video frame")
    parser.add_argument("--out", help="output HTML (default: evaluation/reports/viewer_<pred>.html)")
    args = parser.parse_args()

    if args.end < args.start:
        parser.error("--end is before --start")
    meta_path, raw_path = require(["atd_meta", "atd_raw"])
    team = team_of_each_track(meta_path).tolist()

    pred = read_labels(args.pred)
    gt = read_labels(args.gt) if args.gt != "none" else np.full(len(pred), "X")
    window = slice(args.start, args.end + 1)

    payload = {
        "start": args.start,
        "pitch": list(GAMES[GAME]["pitch"]),
        "team": team,
        "pred": "".join(pred[window]),
        "gt": "".join(gt[window]),
        "name": os.path.basename(args.pred),
        "frames": read_window(raw_path, args.start, args.end, GAMES[GAME]["pitch"]),
    }

    out = args.out or os.path.join(
        OUT_DIR, "viewer_{}.html".format(os.path.splitext(os.path.basename(args.pred))[0]))
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    with open(out, "w") as handle:
        handle.write(build_html(payload))

    graded = np.array(list(payload["gt"])) != "X"
    wrong = graded & (np.array(list(payload["pred"])) != np.array(list(payload["gt"])))
    print("frames {}..{}  graded {}  wrong {}".format(args.start, args.end, graded.sum(), wrong.sum()))
    print("wrote", os.path.relpath(out, ROOT), "-- open it in a browser. Do not commit or share it.")


if __name__ == "__main__":
    main()
