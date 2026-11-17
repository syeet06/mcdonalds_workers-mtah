"""Is the tracked ball on a ball in the picture? White pixels in the square around it, per frame.

    from pipeline.common import load_stage
    white = load_stage("ball_seen")["white"]     # (N,) int16, -1 where there is no square to look in

The tracking ball is projected into the tactical video through its homography, as the viewer draws
it, and the square the viewer draws around it (SQUARE_PX across) is searched for bright, colourless
pixels: the ball. Watching our invented possessions on the viewer, the square is empty on most of
them: the tracking has put the ball where there is none. A round-blob detector (ball_video_labels.
snap_ball) can't be used to say so, because at a player's feet the ball merges with a white boot or
sock into one long blob and is missed; it finds a blob on only 43% of the frames of real
possessions. Counting white pixels doesn't need a clean blob. decoder.merge_ballless_spells reads it.

Needs PyAV (requirements-video.txt) the first time: the whole tactical video is read once (~3 min),
then the result is cached in pipeline/cache/ball_seen.npz like the other stages.
"""

import os

import numpy as np

from pipeline.common import N_FRAMES, PITCH, half_of_frame, load_tracking

# Params
SQUARE_PX = 34           # the viewer's ball square, image pixels across (1920 x 1080 video)
WHITE_V = 210            # HSV value at least this...
WHITE_S = 45             # ...and saturation at most this: white
def tactical_video():
    """The tactical video's path, from ./data or MTAH_DATA_DIR (games.require), like every other asset."""
    from games import require
    return require(["video"])[0]


def ball_pixels(ball, H, pitch=PITCH):
    """(N, 2) float image pixels of pitch-metre points through per-frame homographies H (normalised
    0-1 pitch coordinates to pixels); NaN where either is missing or the point projects to nonsense."""
    x, y = ball[:, 0] / pitch[0], ball[:, 1] / pitch[1]
    den = H[:, 2, 0] * x + H[:, 2, 1] * y + H[:, 2, 2]
    with np.errstate(invalid="ignore", divide="ignore"):
        u = (H[:, 0, 0] * x + H[:, 0, 1] * y + H[:, 0, 2]) / den
        v = (H[:, 1, 0] * x + H[:, 1, 1] * y + H[:, 1, 2]) / den
    px = np.stack([u, v], axis=1)
    px[~(np.isfinite(px).all(axis=1) & (np.abs(px) < 30000).all(axis=1))] = np.nan
    return px


def white_in_square(img, u, v, square=SQUARE_PX):
    """Number of white pixels (WHITE_V, WHITE_S) in the square of side `square` centred on (u, v), or
    -1 if the square doesn't fit inside the image."""
    import cv2
    h = square // 2
    x, y = int(round(u)), int(round(v))
    if not (h <= x <= img.shape[1] - h and h <= y <= img.shape[0] - h):
        return -1
    hsv = cv2.cvtColor(img[y - h:y + h, x - h:x + h], cv2.COLOR_BGR2HSV)
    return int(((hsv[:, :, 2] >= WHITE_V) & (hsv[:, :, 1] <= WHITE_S)).sum())


def build(video=None):
    """{"white": (N,) int16}: white pixels in the square around the projected tracking ball, on every
    graded frame with a tracking ball and a homography; -1 elsewhere."""
    import av
    from starter.load_homography import load

    H, _, _ = load()
    px = ball_pixels(load_tracking()["ball"].astype(float), H)
    want = (half_of_frame() > 0) & np.isfinite(px[:, 0])
    white = np.full(N_FRAMES, -1, dtype=np.int16)
    from pipeline import progress
    bar = progress.Bar(N_FRAMES, "reading the video for the ball square")
    with av.open(video or tactical_video()) as container:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        for frame in container.decode(stream):
            f = round(float(frame.pts * stream.time_base) * 25)
            if 0 <= f < N_FRAMES and want[f]:
                white[f] = white_in_square(frame.to_ndarray(format="bgr24"), px[f, 0], px[f, 1])
            bar.update(f + 1)
    bar.close()
    return {"white": white}
