"""Training data for the WASB fine-tune: samples, hard negatives, a JPEG tile cache and a Dataset.

    from pipeline import ball_video_data as d
    d.build()                                       # AdaBN without the held-out fold, mining, cache
    ds = d.TileDataset(samples, d.CACHE_DIR, donors=occ.load_donor_bank())

A sample is three consecutive frames cropped to one 640x360 window (tx, ty), with the ball at (bx, by)
inside the window in source pixels (NaN when the window has no ball). The cache stores the three
crops, resized to 512x288, side by side in one JPEG.
"""

import os

import cv2
import numpy as np
import pandas as pd
import torch

from pipeline.ball_video_io import FRAME_WH, IN_H, IN_W, TILE_H, TILE_W, prep
from pipeline.common import CACHE_DIR as PIPELINE_CACHE

# Params
HELD_OUT = 4                        # fold kept out of training, AdaBN and mining; evaluation only
CACHE_DIR = os.path.join(PIPELINE_CACHE, "ball_video", "tiles")
BALL_MARGIN = 16                    # px; positives keep the ball this far inside the window
NEG_CLEARANCE = 40                  # px; negatives keep the ball this far outside the window
HARD_PEAK = 0.3                     # mined blobs at least this confident...
HARD_MISS = 40                      # ...and this far from the ball are hard negatives
HARD_MAX = 5000
SIGMA = 2.5                         # target disc radius in model pixels (WASB binary_fixed_size)
P_OCCLUDE = 0.4
JITTER = 0.15                       # brightness and contrast, same for the 3 frames
JPEG_QUALITY = 95
SCALE = IN_W / TILE_W               # source px -> model px


def binary_target(bx, by, sigma=SIGMA):
    """(288, 512) float32: 1 within `sigma` of (bx, by) in model pixels, 0 elsewhere; all 0 for NaN."""
    t = np.zeros((IN_H, IN_W), np.float32)
    if np.isnan(bx) or np.isnan(by):
        return t
    ys, xs = np.ogrid[:IN_H, :IN_W]
    t[(xs - bx) ** 2 + (ys - by) ** 2 <= sigma ** 2] = 1
    return t


def _window_for(x, y, rng):
    """A window containing (x, y) at a uniform position, BALL_MARGIN inside, clamped to the frame."""
    tx = int(np.clip(round(x - rng.uniform(BALL_MARGIN, TILE_W - BALL_MARGIN)), 0, FRAME_WH[0] - TILE_W))
    ty = int(np.clip(round(y - rng.uniform(BALL_MARGIN, TILE_H - BALL_MARGIN)), 0, FRAME_WH[1] - TILE_H))
    return tx, ty


def _clear_of(x, y, tx, ty, clearance=NEG_CLEARANCE):
    return not (tx - clearance < x < tx + TILE_W + clearance and ty - clearance < y < ty + TILE_H + clearance)


def select_samples(labels, n_pos=10000, n_rand_neg=5000, seed=0):
    """Positives (ball in the window) and random negatives (ball well outside it), training folds only,
    from snapped labels."""
    rng = np.random.default_rng(seed)
    train = labels[labels["snapped"] & (labels["fold"] != HELD_OUT)]
    inner = train[train["x"].between(BALL_MARGIN, FRAME_WH[0] - BALL_MARGIN, inclusive="left")
                  & train["y"].between(BALL_MARGIN, FRAME_WH[1] - BALL_MARGIN, inclusive="left")]
    rows = []
    for r in inner.iloc[rng.choice(len(inner), min(n_pos, len(inner)), replace=False)].itertuples():
        tx, ty = _window_for(r.x, r.y, rng)
        rows.append((r.frame, tx, ty, "pos", r.x - tx, r.y - ty, r.ppm, r.fold))
    for r in train.iloc[rng.choice(len(train), min(n_rand_neg, len(train)), replace=False)].itertuples():
        for _ in range(20):
            tx = int(rng.integers(0, FRAME_WH[0] - TILE_W + 1))
            ty = int(rng.integers(0, FRAME_WH[1] - TILE_H + 1))
            if _clear_of(r.x, r.y, tx, ty):
                rows.append((r.frame, tx, ty, "neg", np.nan, np.nan, r.ppm, r.fold))
                break
    return pd.DataFrame(rows, columns=["frame", "tx", "ty", "kind", "bx", "by", "ppm", "fold"])


def mine_hard_negatives(labels, n_triples=3000, seed=0, max_rows=HARD_MAX):
    """Windows centred on the AdaBN model's confident misses (boots, heads...), training folds only.
    If the ball happens to fall inside such a window, it is labelled there."""
    from games import require
    from pipeline import ball_video as bv, wasb
    from pipeline.ball_video_io import FrameReader
    from starter.load_homography import load

    H, _, _ = load()
    device = wasb.pick_device()
    model = wasb.load_model(bv.ADABN_HELDOUT_PATH, device)
    reader = FrameReader(require(["video"])[0])
    train = labels[labels["snapped"] & (labels["fold"] != HELD_OUT)]
    picks = train.iloc[np.sort(np.random.default_rng(seed).choice(len(train), min(n_triples, len(train)), replace=False))]
    rows = []
    for r in picks.itertuples():
        dets = bv.detect(model, reader.read(int(r.frame), 3), device)[0]
        if len(dets):
            dets = dets[(dets[:, 3] >= HARD_PEAK) & bv.on_pitch(H[int(r.frame)], dets[:, 0], dets[:, 1])]
        for x, y, _, _ in dets:
            if np.hypot(x - r.x, y - r.y) <= HARD_MISS:
                continue
            tx = int(np.clip(round(x - TILE_W / 2), 0, FRAME_WH[0] - TILE_W))
            ty = int(np.clip(round(y - TILE_H / 2), 0, FRAME_WH[1] - TILE_H))
            inside = tx <= r.x < tx + TILE_W and ty <= r.y < ty + TILE_H
            rows.append((r.frame, tx, ty, "hard", r.x - tx if inside else np.nan,
                         r.y - ty if inside else np.nan, r.ppm, r.fold))
        if len(rows) >= max_rows:
            break
    return pd.DataFrame(rows[:max_rows], columns=["frame", "tx", "ty", "kind", "bx", "by", "ppm", "fold"])


def strip_name(r):
    return f"{int(r.frame)}_{int(r.tx)}_{int(r.ty)}.jpg"


def write_cache(samples, reader, cache_dir=CACHE_DIR):
    """One JPEG per sample: its window in frames f, f+1, f+2, each resized to 512x288, side by side."""
    os.makedirs(cache_dir, exist_ok=True)
    samples = samples.sort_values("frame")
    for frame, group in samples.groupby("frame", sort=True):
        imgs = reader.read(int(frame), 3)
        for r in group.itertuples():
            crops = [cv2.resize(img[r.ty:r.ty + TILE_H, r.tx:r.tx + TILE_W], (IN_W, IN_H), interpolation=cv2.INTER_AREA)
                     for img in imgs]
            cv2.imwrite(os.path.join(cache_dir, strip_name(r)), np.hstack(crops), [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
    samples.to_parquet(os.path.join(cache_dir, "samples.parquet"))


class TileDataset(torch.utils.data.Dataset):
    """(x, y): x (9, 288, 512) normalised frames, y (3, 288, 512) ball discs. Positives are occluded
    by a donor player with probability p_occlude, on 1, 2 or 3 of the frames."""

    def __init__(self, samples, cache_dir=CACHE_DIR, donors=None, p_occlude=P_OCCLUDE, jitter=True, seed=0):
        self.samples = samples.reset_index(drop=True)
        self.cache_dir, self.donors, self.p_occlude, self.jitter = cache_dir, donors, p_occlude, jitter
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        from pipeline.occlusion import occlude

        r = self.samples.iloc[i]
        strip = cv2.imread(os.path.join(self.cache_dir, strip_name(r)))
        tiles3 = np.stack(np.split(strip, 3, axis=1))
        bx, by = r.bx * SCALE, r.by * SCALE
        if self.donors and not np.isnan(bx) and self.rng.random() < self.p_occlude:
            which = np.zeros(3, bool)
            which[self.rng.choice(3, self.rng.integers(1, 4), replace=False)] = True
            donor = self.donors[self.rng.integers(len(self.donors))]
            tiles3 = occlude(tiles3, (bx, by), r.ppm, donor, which, self.rng)
        if self.jitter:
            c, b = 1 + self.rng.uniform(-JITTER, JITTER), 255 * self.rng.uniform(-JITTER, JITTER)
            tiles3 = np.clip(tiles3.astype(np.float32) * c + b, 0, 255).astype(np.uint8)
        x = np.concatenate([prep(t) for t in tiles3])
        y = np.repeat(binary_target(bx, by)[None], 3, axis=0)
        return torch.from_numpy(x), torch.from_numpy(y)


def build(seed=0):
    """AdaBN without the held-out fold, mined hard negatives, sampled positives and negatives, cache."""
    from games import require
    from pipeline import ball_video as bv
    from pipeline.ball_video_io import FrameReader
    from pipeline.ball_video_labels import LABELS_PATH

    bv.make_adabn(exclude_fold=HELD_OUT, out_path=bv.ADABN_HELDOUT_PATH)
    labels = pd.read_parquet(LABELS_PATH)
    samples = pd.concat([select_samples(labels, seed=seed), mine_hard_negatives(labels, seed=seed)], ignore_index=True)
    samples = samples.drop_duplicates(["frame", "tx", "ty"], ignore_index=True)    # two misses, one window
    write_cache(samples, FrameReader(require(["video"])[0]))
    return samples
