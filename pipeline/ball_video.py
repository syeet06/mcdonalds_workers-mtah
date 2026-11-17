"""Find the ball in the tactical video with WASB: detection, AdaBN adaptation and scoring.

    from pipeline import ball_video as bv, wasb
    model = wasb.load_model(bv.make_adabn(), wasb.pick_device())
    dets = bv.detect(model, frames, device)                   # per frame (k, 4): x, y, score, peak
    xyp = bv.top_on_pitch(dets[0], H[f], min_peak=0.5)        # strongest on-pitch blob, or NaN

Scores are measured against the tracking ball projected into the image, on frames where the tracking
has it: a hit is within 40 px (the projection itself is off by 5-35 px).
"""

import hashlib
import os
import time

import numpy as np
import pandas as pd
import torch

from pipeline.ball_video_io import IN_W, TILE_H, TILE_W, blobs, on_pitch, prep, tile_grid, to_pitch
from pipeline.common import BALL_VIDEO_PATH, CACHE_DIR, WEIGHTS_DIR

# Params
BATCH_TILES = 8                     # 16 tiles at once overflow 8 GB on the M1 and swap
DETECT_THR = 0.2                    # blobs are kept down to this heatmap value; filter on peak later
HIT_RADIUS = 40                     # px
ADABN_TRIPLES = 300
ADABN_PATH = os.path.join(CACHE_DIR, "ball_video", "wasb_adabn_640.pt")
ADABN_HELDOUT_PATH = os.path.join(CACHE_DIR, "ball_video", "wasb_adabn_640_no_fold4.pt")
SPIKE_WINDOWS = (20400, 65750, 69500, 80750, 163500, 168000, 179250)
SPIKE_WINDOW = 375
EVAL_PEAKS = (0.5, 0.2)
EVAL_MISSING = (0.3, 0.6)           # windows with this share of missing ball, like the spike's
# The fine-tuned weights used for the submission, committed (organisers confirmed model weights are fine,
# 2026-11-08). ball_video_train.train() writes to the cache; copy a new best checkpoint here to adopt it.
FINETUNED_WEIGHTS = os.path.join(WEIGHTS_DIR, "ball_video_finetuned.pt")
RUN_CHUNK = 150                     # frames read at once (~1 GB of 1080p), a multiple of 3
BLOCKS_PATH = os.path.join(CACHE_DIR, "ball_video", "blocks_detections.npz")
BUILD_PATH = os.path.join(CACHE_DIR, "ball_video", "gap_detections.npz")
GAP_SHEET_PATH = os.path.join(CACHE_DIR, "ball_video", "gap_sheet.jpg")


def _forward(model, triple, device):
    """(16, 3, 288, 512) heatmaps (sigmoid) for one triple of frames, all tiles."""
    batch = np.stack([np.concatenate([prep(img[y:y + TILE_H, x:x + TILE_W]) for img in triple])
                      for x, y in tile_grid()])
    with torch.no_grad():
        return np.concatenate([torch.sigmoid(model(torch.from_numpy(batch[k:k + BATCH_TILES]).to(device))[0])
                               .cpu().numpy() for k in range(0, len(batch), BATCH_TILES)])


def detect(model, frames, device, thr=DETECT_THR):
    """Blobs per frame, (k, 4) of x, y, score, peak in source pixels. Frames go in disjoint triples;
    a leftover 1-2 frames at the end reuse the last full triple."""
    n = len(frames)
    starts = list(range(0, n - 2, 3)) + ([n - 3] if n % 3 else [])
    out = [None] * n
    for s in starts:
        hms = _forward(model, frames[s:s + 3], device)
        for j in range(3):
            if out[s + j] is None:
                out[s + j] = _to_source(hms[:, j], thr)
    return out


def _to_source(hms, thr):
    """Blobs of one frame's 16 tile heatmaps, in source pixels."""
    parts = []
    for (x, y), hm in zip(tile_grid(), hms):
        b = blobs(hm, thr)
        if len(b):
            b[:, 0] = x + b[:, 0] * TILE_W / IN_W
            b[:, 1] = y + b[:, 1] * TILE_W / IN_W
            parts.append(b)
    return np.concatenate(parts) if parts else np.zeros((0, 4))


def top_on_pitch(dets, H, min_peak):
    """(3,) x, y, peak of the highest-scoring on-pitch blob with peak >= min_peak, else NaNs."""
    if len(dets):
        dets = dets[(dets[:, 3] >= min_peak) & on_pitch(H, dets[:, 0], dets[:, 1])]
    if not len(dets):
        return np.full(3, np.nan)
    best = dets[np.argmax(dets[:, 2])]
    return best[[0, 1, 3]]


def score(est_xy, ref_uv, radius=HIT_RADIUS):
    """On frames with a reference: share with an output, share with an output within `radius`, and
    hits per output."""
    est_xy, ref_uv = np.asarray(est_xy, float), np.asarray(ref_uv, float)
    ref = ~np.isnan(ref_uv[:, 0])
    has = ~np.isnan(est_xy[ref, 0])
    with np.errstate(invalid="ignore"):
        hit = np.hypot(*(est_xy[ref] - ref_uv[ref]).T) < radius
    return {"output": float(has.mean()), "hit": float(hit.mean()), "precision": float(hit.sum() / max(has.sum(), 1))}


def adapt_bn(model, reader, frames, device):
    """AdaBN: re-estimate BatchNorm running statistics on our video (no labels). Each start frame in
    `frames` gives one triple, all 16 tiles. Returns the model in eval mode."""
    model.eval()
    for m in model.modules():
        if isinstance(m, torch.nn.BatchNorm2d):
            m.reset_running_stats()
            m.momentum = None                   # cumulative average over every batch
            m.train()
    model.to(device)
    for f in frames:
        _forward(model, reader.read(int(f), 3), device)
    return model.eval()


def make_adabn(n=ADABN_TRIPLES, seed=0, exclude_fold=None, out_path=ADABN_PATH):
    """AdaBN weights from `n` random graded triples (outside `exclude_fold`), saved to out_path."""
    from games import require
    from pipeline import wasb
    from pipeline.ball_impute import chunk_fold
    from pipeline.ball_video_io import FrameReader
    from pipeline.common import HALVES, half_of_frame

    if os.path.exists(out_path):
        return out_path
    pool = np.concatenate([np.arange(first, last - 3) for first, last in HALVES])
    if exclude_fold is not None:
        pool = pool[chunk_fold(pool, half_of_frame()) != exclude_fold]
    frames = np.sort(np.random.default_rng(seed).choice(pool, n, replace=False))
    device = wasb.pick_device()
    model = wasb.load_model(wasb.fetch_pretrained(wasb.PRETRAINED_PATH), device)
    model = adapt_bn(model, FrameReader(require(["video"])[0]), frames, device)
    torch.save({k: v.cpu() for k, v in model.state_dict().items()}, out_path)
    return out_path


def reproduce_spike(min_peak=0.5):
    """The spike's 7 windows x 375 frames with AdaBN weights: visible hit/precision, gap output."""
    from games import require
    from pipeline import wasb
    from pipeline.ball_video_io import FrameReader
    from pipeline.common import PITCH, load_tracking
    from starter.load_homography import load, project

    trk = load_tracking()
    H, _, _ = load()
    device = wasb.pick_device()
    model = wasb.load_model(make_adabn(), device)
    reader = FrameReader(require(["video"])[0])
    est, ref = [], []
    for start in SPIKE_WINDOWS:
        for f, d in zip(range(start, start + SPIKE_WINDOW), detect(model, reader.read(start, SPIKE_WINDOW), device)):
            est.append(top_on_pitch(d, H[f], min_peak)[:2])
            ref.append(project(H[f], *(trk["ball"][f] / np.array(PITCH))))
    est, ref = np.array(est), np.array(ref, float)
    out = score(est, ref)
    out["gap_output"] = float(np.mean(~np.isnan(est[np.isnan(ref[:, 0]), 0])))
    return out


def subset_table(est, ref, masks):
    """Rows all / each mask / gap; columns output, hit, precision per peak threshold. `est` maps a
    peak threshold to (N, 2) estimates; `ref` is (N, 2), NaN on gap frames. Gap gets output only."""
    gap = np.isnan(ref[:, 0])
    rows = {"all": ~gap, **{k: m & ~gap for k, m in masks.items()}}
    table = {}
    for name, m in rows.items():
        table[name] = {}
        for p, e in est.items():
            r = score(e[m], ref[m]) if m.any() else dict.fromkeys(["output", "hit", "precision"], np.nan)
            table[name].update({f"{k}_{p}": v for k, v in r.items()})
    table["gap"] = {f"{k}_{p}": (float(np.mean(~np.isnan(e[gap, 0]))) if k == "output" and gap.any() else np.nan)
                    for p, e in est.items() for k in ["output", "hit", "precision"]}
    return pd.DataFrame.from_dict(table, orient="index")


def eval_windows(n_windows, window, seed=0):
    """Start frames of `n_windows` non-overlapping held-out windows with EVAL_MISSING missing ball."""
    from pipeline.ball_impute import chunk_fold
    from pipeline.ball_video_data import HELD_OUT
    from pipeline.common import half_of_frame, load_tracking

    half = half_of_frame()
    missing = np.isnan(load_tracking()["ball"][:, 0])
    ok = (half > 0) & (chunk_fold(np.arange(len(half)), half) == HELD_OUT)
    run = np.convolve(ok, np.ones(window, int), "valid") == window
    share = np.convolve(missing, np.ones(window), "valid") / window
    cand = np.flatnonzero(run & (share >= EVAL_MISSING[0]) & (share <= EVAL_MISSING[1]))
    picked = []
    for s in np.random.default_rng(seed).permutation(cand):
        if all(abs(s - q) >= window for q in picked):
            picked.append(int(s))
        if len(picked) == n_windows:
            break
    return sorted(picked)


def evaluate(weights, n_windows=10, window=SPIKE_WINDOW, seed=0, sheet_path=GAP_SHEET_PATH):
    """Held-out comparison of models (name -> weights path): subset_table per model, stacked. Each
    window is read once for all models. The last model's confident gap detections go on a contact sheet."""
    import cv2

    from games import require
    from pipeline import wasb
    from pipeline.ball_video_io import FrameReader
    from pipeline.ball_video_labels import gap_like
    from pipeline.common import PITCH, load_tracking
    from starter.load_homography import load, project

    trk = load_tracking()
    H, _, _ = load()
    device = wasb.pick_device()
    models = {name: wasb.load_model(path, device) for name, path in weights.items()}
    reader = FrameReader(require(["video"])[0])
    starts = eval_windows(n_windows, window, seed)
    frames = np.concatenate([np.arange(s, s + window) for s in starts])
    est = {name: {p: [] for p in EVAL_PEAKS} for name in models}
    crops, rng = [], np.random.default_rng(seed)
    for s in starts:
        imgs = reader.read(s, window)
        for name, model in models.items():
            for f, img, d in zip(range(s, s + window), imgs, detect(model, imgs, device)):
                for p in EVAL_PEAKS:
                    est[name][p].append(top_on_pitch(d, H[f], p)[:2])
                x, y = est[name][EVAL_PEAKS[0]][-1]
                if name == list(models)[-1] and np.isnan(trk["ball"][f, 0]) and not np.isnan(x) and rng.random() < 0.05:
                    crops.append((f, _crop(img, x, y)))
        print(f"window {s} done", flush=True)
    ref = np.array([project(H[f], *(trk["ball"][f] / np.array(PITCH))) for f in frames], float)
    masks = {k: m[frames] for k, m in gap_like(trk).items()}
    table = pd.concat({name: subset_table({p: np.array(v) for p, v in e.items()}, ref, masks)
                       for name, e in est.items()})
    if crops:
        picks = [crops[i] for i in sorted(rng.choice(len(crops), min(24, len(crops)), replace=False))]
        tiles = [cv2.putText(c.copy(), str(f), (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1) for f, c in picks]
        tiles += [np.zeros_like(tiles[0])] * (-len(tiles) % 6)
        cv2.imwrite(sheet_path, np.vstack([np.hstack(tiles[i:i + 6]) for i in range(0, len(tiles), 6)]))
    return table


def _crop(img, x, y, half=80):
    """A 160x160 crop centred on (x, y), padded at the frame edge, with a ring at the centre."""
    import cv2

    pad = cv2.copyMakeBorder(img, half, half, half, half, cv2.BORDER_CONSTANT)
    c = pad[int(y):int(y) + 2 * half, int(x):int(x) + 2 * half].copy()
    return cv2.circle(c, (half, half), 12, (0, 0, 255), 1)


def pitch_metres(Hs, est):
    """(N, 2) pitch metres of image estimates `est` (N, >=2: u, v, ...) through each frame's
    homography `Hs` (N, 3, 3); NaN where there is no estimate."""
    from pipeline.common import PITCH

    out = np.full((len(est), 2), np.nan)
    for i in np.flatnonzero(~np.isnan(est[:, 0])):
        out[i] = np.array(to_pitch(Hs[i], est[i, 0], est[i, 1])) * np.array(PITCH)
    return out


def run_key(weights, min_peak):
    """Identity of a detection run: the weights file's content and the peak threshold."""
    with open(weights, "rb") as f:
        return f"{hashlib.sha256(f.read()).hexdigest()} {min_peak}"


def _resume(cache_path, frames, key):
    """(est, done) saved by an earlier run of the same frames, weights and threshold, else empty."""
    est = np.full((len(frames), 3), np.nan)
    done = np.zeros(len(frames), bool)
    if os.path.exists(cache_path):
        with np.load(cache_path) as c:
            if np.array_equal(c["frames"], frames) and "key" in c.files and str(c["key"]) == key:
                est, done = c["est"], c["done"]
    return est, done


def ball_pitch(frames, weights, min_peak=EVAL_PEAKS[0], cache_path=BLOCKS_PATH, device=None):
    """(metres (N, 2), peak (N,)) of the top on-pitch detection on each of `frames` (sorted), NaN
    without one. Runs in RUN_CHUNK pieces and saves after each, so an interrupted run resumes; a cache
    from other frames, weights or `min_peak` is started over. `device` defaults to wasb.pick_device()."""
    from games import require
    from pipeline import wasb
    from pipeline.ball_video_io import FrameReader
    from starter.load_homography import load

    frames = np.asarray(frames)
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)   # a fresh clone has no pipeline/cache/ball_video/
    key = run_key(weights, min_peak)
    est, done = _resume(cache_path, frames, key)
    H, _, _ = load()
    device = device or wasb.pick_device()
    model = wasb.load_model(weights, device)
    reader = FrameReader(require(["video"])[0])
    breaks = np.flatnonzero(np.diff(frames) != 1) + 1
    pieces = [r[k:k + RUN_CHUNK] for r in np.split(np.arange(len(frames)), breaks) for k in range(0, len(r), RUN_CHUNK)]
    t0, done0 = time.monotonic(), int(done.sum())
    print(f"{len(frames)} frames in {len(pieces)} pieces, {done0} already done (resumed)" if done0
          else f"{len(frames)} frames in {len(pieces)} pieces", flush=True)
    for n, idx in enumerate(pieces):
        if done[idx].all():
            continue
        f0 = int(frames[idx[0]])
        imgs = reader.read(f0, max(len(idx), 3))            # a 1-2 frame piece borrows the next frames
        for i, d in zip(idx, detect(model, imgs, device)):
            est[i] = top_on_pitch(d, H[int(frames[i])], min_peak)
        done[idx] = True
        np.savez(cache_path + ".tmp.npz", frames=frames, est=est, done=done, key=key)
        os.replace(cache_path + ".tmp.npz", cache_path)          # a kill mid-save can't corrupt the cache
        rate = (int(done.sum()) - done0) / max(time.monotonic() - t0, 1e-9)       # frames/s this run
        left = (len(frames) - int(done.sum())) / rate / 3600 if rate else float("nan")
        print(f"piece {n + 1}/{len(pieces)} frame {f0} done {done.mean():.1%} · {rate:.1f} frames/s · ~{left:.1f} h left",
              flush=True)
    return pitch_metres(H[frames], est), est[:, 2]


def build(weights=None, out_path=BALL_VIDEO_PATH, device=None):
    """Fine-tuned detections on every missing-ball graded frame: {"ball_video": (N, 2) metres,
    "ball_video_peak": (N,)}, NaN elsewhere, also saved to `out_path`. About 11 h on MPS; resumable."""
    from pipeline import common

    trk = common.load_tracking()
    frames = np.flatnonzero(common.graded_mask() & np.isnan(trk["ball"][:, 0]))
    xy, peak = ball_pitch(frames, weights or FINETUNED_WEIGHTS, cache_path=BUILD_PATH, device=device)
    out = {"ball_video": np.full((len(trk["ball"]), 2), np.nan), "ball_video_peak": np.full(len(trk["ball"]), np.nan)}
    out["ball_video"][frames], out["ball_video_peak"][frames] = xy, peak
    np.savez(out_path, **out)
    return out


def decoder_test(weights=FINETUNED_WEIGHTS, ws=(0, 0.5, 1, 2), device=None):
    """The adoption test: video detections on the hidden-ball blocks as the decoder's inferred-ball
    clue, overall agreement per weight (0 is the 0.763 baseline; adopted at 2 -> 0.809). ~10.5 h on
    MPS for detection (cached in BLOCKS_PATH), then minutes per weight."""
    from pipeline import ball_filled as bf, ball_impute as bi, clues, decoder as dc
    from pipeline.common import half_of_frame, load_tracking
    from pipeline.gap_validation import evaluate_decoder

    trk = load_tracking()
    half = half_of_frame()
    ball, _ = bf.remove_teleports(trk["ball"])
    ball, _ = bf.fill_short_gaps(ball, half, 25)
    trk = dict(trk, ball=ball)
    starts, ends = bi.fake_gaps(trk, half)
    frames = np.concatenate([np.arange(s, e) for s, e in zip(starts, ends)])
    video = np.full(ball.shape, np.nan)
    video[frames] = ball_pitch(frames, weights, cache_path=BLOCKS_PATH, device=device)[0]
    intervals = clues.tag_intervals()
    rows = []
    for w in ws:
        run = dc.harness_run(trk, half, intervals, p_switch=16, w_tag=0.5,
                             extra_fn=lambda hidden, w=w: clues.inferred_ball_clue(video, trk["players"], trk["team"], w))
        tab = evaluate_decoder(trk, run).set_index("bucket")
        rows.append({"w_video": w, "overall": (tab["agreement"] * tab["blocks"]).sum() / tab["blocks"].sum(),
                     **tab["agreement"].to_dict()})
    return pd.DataFrame(rows)
