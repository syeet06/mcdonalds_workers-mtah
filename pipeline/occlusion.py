"""Synthetic occlusion for the WASB fine-tune: paste a real player over the ball, keep the label.

    from pipeline import occlusion as occ
    donors = occ.build_donor_bank()                          # cut-out players from our own frames
    tiles3 = occ.occlude(tiles3, ball_xy, ball_ppm, donors[0], [False, True, True], rng)

The labels exist only where the ball is visible; gaps are mostly contested or occluded balls. Pasting
players over visible balls gives the model that view with a known answer.
"""

import os

import cv2
import numpy as np

from pipeline.ball_video_io import IN_W, TILE_W
from pipeline.common import CACHE_DIR, PITCH

# Params
GRASS_DISTANCE = 12.0              # Lab distance from the local grass above which a pixel is "player"
LIGHTNESS_WEIGHT = 0.4             # mowing stripes differ in lightness, players in colour
WINDOW_W, WINDOW_H, FEET_ABOVE_BOTTOM = 3.0, 4.0, 1.2   # cut window, in pixels per metre
MIN_BODY_AREA = 0.3                # blob area / ppm^2 to count as a player
ASPECT, HEIGHT_PPM, FILL = (1.3, 4.5), (1.0, 3.2), (0.3, 0.95)   # plausible player cut-out
MAX_GREEN = 0.6                    # share of a cut-out's pixels that may look like grass
DONOR_MIN_BALL_M, DONOR_MIN_OTHERS_M = 10.0, 3.0
FEET_DX, FEET_DY = (-0.2, 0.3), (0.1, 0.4)   # donor feet relative to the ball, in ppm at tile scale
DONORS_PATH = os.path.join(CACHE_DIR, "ball_video", "donors.npz")
DONORS_SHEET = os.path.join(CACHE_DIR, "ball_video", "donors.jpg")


def cut_player(img, feet_xy, ppm, threshold=GRASS_DISTANCE):
    """(crop, alpha) of the player standing at feet_xy, or None if the cut window leaves the image.

    A pixel is "player" when its colour is far from the local grass: the median of the window's
    border in Lab, lightness down-weighted. Hue keying loses heads and legs at this size. The window
    is generous because the projected feet are off by up to ~35 px; the cut-out is trimmed to the
    player, so its bottom is the feet."""
    w, h = int(WINDOW_W * ppm), int(WINDOW_H * ppm)
    x0, y0 = int(feet_xy[0]) - w // 2, int(feet_xy[1]) - h + int(FEET_ABOVE_BOTTOM * ppm)
    if x0 < 0 or y0 < 0 or x0 + w > img.shape[1] or y0 + h > img.shape[0]:
        return None
    crop = img[y0:y0 + h, x0:x0 + w].copy()
    lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB).astype(float)
    border = np.concatenate([lab[:3].reshape(-1, 3), lab[-3:].reshape(-1, 3),
                             lab[:, :3].reshape(-1, 3), lab[:, -3:].reshape(-1, 3)])
    grass = np.median(border, axis=0)
    dist = np.sqrt((LIGHTNESS_WEIGHT * (lab[..., 0] - grass[0])) ** 2 + ((lab[..., 1:] - grass[1:]) ** 2).sum(-1))
    body = cv2.morphologyEx((dist > threshold).astype(np.uint8), cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    n, ids, stats, cent = cv2.connectedComponentsWithStats(body)
    if n < 2:
        return None
    # The player: of the sizeable blobs, the one nearest the expected feet position.
    feet = np.array([w / 2, h - FEET_ABOVE_BOTTOM * ppm])
    big = [k for k in range(1, n) if stats[k, cv2.CC_STAT_AREA] > MIN_BODY_AREA * ppm ** 2] or list(range(1, n))
    k = min(big, key=lambda k: np.hypot(*(cent[k] - feet)))
    x, y, bw, bh, _ = stats[k]
    alpha = cv2.GaussianBlur((ids == k).astype(np.float32), (3, 3), 0)
    return crop[y:y + bh, x:x + bw], alpha[y:y + bh, x:x + bw]


def plausible_player(crop, alpha, ppm):
    """Whether a cut-out looks like one player: taller than wide, a player's height for the scale,
    reasonably filled, and not mostly grass. Rejects mowing-stripe patches and line paint."""
    h, w = alpha.shape
    keep = alpha > 0.5
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)[keep]
    green = ((hsv[:, 0] >= 30) & (hsv[:, 0] <= 90) & (hsv[:, 1] > 50)).mean() if len(hsv) else 1.0
    return bool(ASPECT[0] <= h / w <= ASPECT[1] and HEIGHT_PPM[0] <= h / ppm <= HEIGHT_PPM[1]
                and FILL[0] <= keep.mean() <= FILL[1] and green < MAX_GREEN)


def paste(img, crop, alpha, feet_xy):
    """img with crop blended in through alpha, its bottom-centre at feet_xy; clipped at the borders."""
    out = img.copy()
    h, w = alpha.shape
    x0, y0 = int(round(feet_xy[0])) - w // 2, int(round(feet_xy[1])) - h
    ix0, iy0 = max(x0, 0), max(y0, 0)
    ix1, iy1 = min(x0 + w, img.shape[1]), min(y0 + h, img.shape[0])
    if ix0 >= ix1 or iy0 >= iy1:
        return out
    a = alpha[iy0 - y0:iy1 - y0, ix0 - x0:ix1 - x0, None]
    c = crop[iy0 - y0:iy1 - y0, ix0 - x0:ix1 - x0]
    region = out[iy0:iy1, ix0:ix1].astype(float)
    out[iy0:iy1, ix0:ix1] = (a * c + (1 - a) * region).astype(np.uint8)
    return out


def occlude(tiles3, ball_xy, ball_ppm, donor, which, rng):
    """Paste `donor` over the ball on the frames where `which` is True. tiles3 is (3, 288, 512, 3)
    uint8 in tile pixels; ball_ppm is pixels per metre at the ball in source pixels."""
    scale = ball_ppm / donor["ppm"] * IN_W / TILE_W
    size = (max(1, round(donor["crop"].shape[1] * scale)), max(1, round(donor["crop"].shape[0] * scale)))
    crop = cv2.resize(donor["crop"], size, interpolation=cv2.INTER_AREA)
    alpha = cv2.resize(donor["alpha"].astype(np.float32), size, interpolation=cv2.INTER_AREA)
    ppm_tile = ball_ppm * IN_W / TILE_W
    feet = (ball_xy[0] + rng.uniform(*FEET_DX) * ppm_tile, ball_xy[1] + rng.uniform(*FEET_DY) * ppm_tile)
    out = tiles3.copy()
    for j, on in enumerate(which):
        if on:
            out[j] = paste(tiles3[j], crop, alpha, feet)
    return out


def build_donor_bank(n=300, seed=0):
    """Cut `n` players standing alone, far from the ball, from graded frames across the match. Saves
    DONORS_PATH and a contact sheet at DONORS_SHEET for a by-eye check."""
    from games import require
    from pipeline.ball_video_io import FrameReader
    from pipeline.ball_video_labels import px_per_metre
    from pipeline.common import graded_mask, load_tracking
    from starter.load_homography import load, project

    trk = load_tracking()
    H, _, _ = load()
    reader = FrameReader(require(["video"])[0])
    rng = np.random.default_rng(seed)
    candidates = np.flatnonzero(graded_mask() & ~np.isnan(trk["ball"][:, 0]))
    donors = []
    for f in np.sort(rng.choice(candidates, 4 * n, replace=False)):
        p = trk["players"][f]
        with np.errstate(invalid="ignore"):
            far = np.linalg.norm(p - trk["ball"][f], axis=1) > DONOR_MIN_BALL_M
            others = np.linalg.norm(p[:, None] - p[None], axis=2)
            np.fill_diagonal(others, np.inf)
            alone = np.nanmin(others, axis=1) > DONOR_MIN_OTHERS_M
        ok = np.flatnonzero(far & alone & ~np.isnan(p[:, 0]))
        if not len(ok):
            continue
        s = int(rng.choice(ok))
        xy_n = p[s] / np.array(PITCH)
        ppm = px_per_metre(H[f], xy_n)
        cut = cut_player(reader.read(int(f), 1)[0], project(H[f], *xy_n), ppm)
        if cut is not None and plausible_player(*cut, ppm):
            donors.append({"crop": cut[0], "alpha": cut[1], "ppm": ppm})
        if len(donors) == n:
            break
    save_donor_bank(donors)
    return donors


def save_donor_bank(donors, path=DONORS_PATH, sheet=DONORS_SHEET):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    arrays = {"ppm": np.array([d["ppm"] for d in donors])}
    for i, d in enumerate(donors):
        arrays[f"crop_{i}"], arrays[f"alpha_{i}"] = d["crop"], d["alpha"]
    np.savez_compressed(path, **arrays)
    cells = []
    for d in donors:
        cell = np.full((80, 40, 3), 255, np.uint8)
        c = (d["crop"] * d["alpha"][..., None] + 255 * (1 - d["alpha"][..., None])).astype(np.uint8)
        c = c[:80, :40]
        cell[80 - c.shape[0]:, (40 - c.shape[1]) // 2:(40 - c.shape[1]) // 2 + c.shape[1]] = c
        cells.append(cell)
    cells += [np.full((80, 40, 3), 255, np.uint8)] * (-len(cells) % 20)
    rows = [np.hstack(cells[r:r + 20]) for r in range(0, len(cells), 20)]
    cv2.imwrite(sheet, cv2.resize(np.vstack(rows), None, fx=2, fy=2, interpolation=cv2.INTER_NEAREST))


def load_donor_bank(path=DONORS_PATH):
    with np.load(path) as z:
        return [{"crop": z[f"crop_{i}"], "alpha": z[f"alpha_{i}"], "ppm": float(ppm)}
                for i, ppm in enumerate(z["ppm"])]
