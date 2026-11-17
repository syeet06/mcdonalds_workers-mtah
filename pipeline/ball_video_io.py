"""Frames, tiles, heatmap blobs and the pitch filter for the WASB ball detector.

    reader = FrameReader(require(["video"])[0])
    frames = reader.read(20400, 3)                          # BGR images
    x = np.concatenate([prep(f[y:y + TILE_H, x:x + TILE_W]) for f in frames])   # (9, 288, 512)

A 1920x1080 frame is cut into 16 overlapping 640x360 tiles, each resized to the model's 512x288, so
the ball (6-7 px) reaches the model at about 5 px, close to the footage it was trained on.
"""

import cv2
import numpy as np

# Params
TILE_W, TILE_H = 640, 360           # tile size in source pixels
IN_W, IN_H = 512, 288               # model input size
TILE_OVERLAP = 0.8                  # tile starts at most 0.8 of a tile apart
FRAME_WH = (1920, 1080)
MEAN = np.array([0.485, 0.456, 0.406])   # ImageNet, RGB, as WASB was trained
STD = np.array([0.229, 0.224, 0.225])
PITCH_MARGIN = 0.02                 # normalised pitch units beyond the lines that still count


def tile_grid(tile_w=TILE_W, frame_wh=FRAME_WH):
    """Top-left corners of the overlapping tiles that cover a frame, row by row."""
    tile_h = tile_w * 9 // 16

    def starts(total, size):
        k = int(np.ceil((total - size) / (TILE_OVERLAP * size))) + 1
        return np.linspace(0, total - size, k).astype(int)

    return [(int(x), int(y)) for y in starts(frame_wh[1], tile_h) for x in starts(frame_wh[0], tile_w)]


def prep(tile_bgr):
    """A BGR uint8 tile -> (3, 288, 512) float32, RGB, ImageNet-normalised."""
    t = cv2.resize(tile_bgr, (IN_W, IN_H), interpolation=cv2.INTER_AREA)[:, :, ::-1] / 255.0
    return ((t - MEAN) / STD).transpose(2, 0, 1).astype(np.float32)


def blobs(hm, thr):
    """Connected regions of a heatmap above thr: (k, 4) of x, y (heatmap-weighted centre), score
    (sum) and peak (max), in heatmap pixels."""
    n, lab = cv2.connectedComponents((hm > thr).astype(np.uint8))
    if n < 2:
        return np.zeros((0, 4))
    lab, w = lab.ravel(), hm.ravel().astype(np.float64)
    ys, xs = np.divmod(np.arange(lab.size), hm.shape[1])
    total = np.bincount(lab, w, n)
    peak = np.zeros(n)
    np.maximum.at(peak, lab, w)
    with np.errstate(invalid="ignore", divide="ignore"):    # label 0 (background) may weigh nothing
        out = np.stack([np.bincount(lab, w * xs, n) / total, np.bincount(lab, w * ys, n) / total, total, peak], 1)
    return out[1:]


def to_pitch(H, x, y):
    """Image pixels -> normalised pitch coordinates (0-1), through the inverse of a frame's homography."""
    p = np.linalg.solve(H, np.stack([np.asarray(x, float), np.asarray(y, float), np.ones_like(x, float)]))
    return p[0] / p[2], p[1] / p[2]


def on_pitch(H, x, y, margin=PITCH_MARGIN):
    """(n,) bool: which image points fall on the pitch, allowing `margin` beyond the lines."""
    px, py = to_pitch(H, x, y)
    return (px >= -margin) & (px <= 1 + margin) & (py >= -margin) & (py <= 1 + margin)


class FrameReader:
    """Reads runs of consecutive frames, seeking only when a run does not follow the last one."""

    def __init__(self, path):
        self._cap = cv2.VideoCapture(path)
        self._next = 0

    def read(self, start, n):
        if start != self._next:
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, start)
        frames = []
        for f in range(start, start + n):
            ok, img = self._cap.read()
            if not ok:
                self._next = -1
                raise ValueError(f"frame {f} could not be read")
            frames.append(img)
        self._next = start + n
        return frames
