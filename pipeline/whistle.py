"""The referee's whistle, from the tactical video's own audio track (same timeline as the labels).

    from pipeline.whistle import whistle_onsets
    onsets = whistle_onsets(half)        # tactical frames where a whistle blast starts

This referee's whistle is the standard steady tones; crowd whistling glides in pitch and is one tone. Per video frame
(128 ms window, 16 kHz mono) the score is the weaker of the two tones' peaks over the 3-5 kHz
median, in dB. A blast is >= WHISTLE_DB for MIN_FRAMES; blasts closer than 1 s are one whistle.

Needs PyAV (pip install av, bundles FFmpeg) for the first run; the score is then cached (~20 s).
"""

import os

import numpy as np

from pipeline.common import CACHE_DIR, FPS, N_FRAMES, runs

# Params
RATE = 16000
HOP = RATE // FPS                 # 640 samples per video frame
NFFT = 2048                       # 128 ms: each tone stays in one 7.8 Hz bin
BAND_HZ = (3000, 5000)            # background for the prominence
TONES_HZ = ((3740, 3880), (4200, 4370))
WHISTLE_DB = 15.0
MIN_FRAMES = 4
MERGE_S = 1.0
CACHE = os.path.join(CACHE_DIR, "whistle.npz")
def tactical_video():
    """The tactical video's path, from ./data or MTAH_DATA_DIR (games.require), like every other asset."""
    from games import require
    return require(["video"])[0]


def decode_mono(path=None):
    """(samples,) float32 mono audio at RATE from the start of the video."""
    import av
    from pipeline import progress
    container = av.open(path or tactical_video())
    resampler = av.AudioResampler(format="s16", layout="mono", rate=RATE)
    bar = progress.Bar(N_FRAMES / FPS, "decoding the audio for the whistle")
    chunks = []
    for frame in container.decode(container.streams.audio[0]):
        chunks.extend(out.to_ndarray().ravel() for out in resampler.resample(frame))
        if frame.time is not None:
            bar.update(frame.time)
    chunks.extend(out.to_ndarray().ravel() for out in resampler.resample(None))
    bar.close()
    return np.concatenate(chunks).astype(np.float32) / 32768


def two_tone_db(audio, n_frames=N_FRAMES):
    """(n_frames,) weaker of the two whistle tones' peak over the band median (dB), centred on each frame."""
    freqs = np.fft.rfftfreq(NFFT, 1 / RATE)
    band = (freqs >= BAND_HZ[0]) & (freqs <= BAND_HZ[1])
    tones = [(freqs[band] >= lo) & (freqs[band] <= hi) for lo, hi in TONES_HZ]
    window = np.hanning(NFFT)
    out = np.zeros(n_frames)
    last = min(n_frames, (len(audio) - NFFT // 2) // HOP)
    for start in range(1, last, 2048):
        idx = np.arange(start, min(start + 2048, last))
        seg = audio[idx[:, None] * HOP - NFFT // 2 + np.arange(NFFT)] * window
        power = np.abs(np.fft.rfft(seg, axis=1))[:, band] ** 2 + 1e-12
        db = 10 * np.log10(power / np.median(power, axis=1, keepdims=True))
        out[idx] = np.minimum(*(db[:, t].max(axis=1) for t in tones))
    return out


def load_score():
    """(N,) two-tone score, computed from the video's audio once and cached."""
    if not os.path.exists(CACHE):
        np.savez(CACHE, score=two_tone_db(decode_mono()))
    with np.load(CACHE) as f:
        return f["score"]


def onsets_from_score(score, half):
    """Frames where a blast starts, graded frames only, blasts closer than MERGE_S merged."""
    blasts = [s for value, s, e in runs((score >= WHISTLE_DB) & (half > 0)) if value and e - s >= MIN_FRAMES]
    keep = []
    for s in blasts:
        if not keep or s - keep[-1] >= MERGE_S * FPS:
            keep.append(s)
    return np.array(keep, dtype=int)


def whistle_onsets(half):
    return onsets_from_score(load_score(), half)
