"""Task 4: which frames are dead-ball frames.

    stage = load_stage("dead_ball")      # runs build() below, cached in pipeline/cache/
    stage["dead"]                        # (N,) bool, True on dead-ball frames

Several independent indicators each vote per frame: -1 dead, +1 live, 0 no opinion. A weighted sum
of the votes below DEAD_THRESHOLD is dead. Short live gaps between stoppages are then filled, and
dead flickers inside live play removed. Exploration, plots and evaluation: task4_dead_ball.ipynb.

Loads Task 3's ball_filled stage. ball_filled.py must never load this stage back.
"""

import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd

from games import require
from pipeline.common import FPS, N_FRAMES, PITCH, load_stage, load_tracking, runs

# Params
OUT_OF_BOUNDS_CONFIG = dict(
    look_ahead_seconds=2,       # a SET PIECES tag may end this long before a long gap's end...
    look_beyond_seconds=4,      # ...or this long after it
)
GOAL_CONFIG = dict(
    distance_from_goal_line=12,     # m: the 4 goals were last seen 7.9, -2.6, 6.9, 9.6 m from the line
    distance_from_goal_centre_y=5,  # m: and 0.7, 12.7, 1.8, 2.9 m from the centre of the goal
    gap_length_seconds=30,          # a goal is followed by a long stoppage
)
STILL_CONFIG = dict(
    median_frames=12 * FPS,     # 12 s rolling median: picks out sustained low-speed spells
    V0=3.0,                     # m/s: at or below this the ball counts as still
    gap_mode='freeze',          # a missing ball holds the stillness count rather than resetting it
    N_still_s=2.0,              # s the ball must be still for a spell to count
)
PLAYER_SLOWDOWN_CONFIG = dict(
    R=60,                       # m: players this close to the ball count towards the average speed
    low=1.0,                    # m/s: average speed at or below this counts as "slowed down"
    high=1.8,                   # m/s: above this, players are "certainly alive"
    steep_decline=-15,          # m/s²: slope below this is a sharp slowdown
    steep_rise=0.5,             # m/s²: slope above this is players speeding up again
    look_back_seconds=15,       # how far before a low period to search for the decline / previous peak
    look_ahead_seconds=5,       # how far after it to search for the speed-up
    min_low_seconds=4,          # how long players must stay slow to count as a stoppage
    peak_tol=0.5,               # m/s: bumps smaller than this don't count as a peak
)
TAGS_CONFIG = dict(
    stoppage_tags=('GOALS', 'SET PIECES'),
    alive_tags=('PROGRESSION', 'DEFENDING IN MIDDLE THIRD', 'ATTACKING TRANSITION', 'DEFENSIVE TRANSITION',
                'LONG BALL', 'BUILD UP', 'DEFENDING IN ATTACKING THIRD'),
)
WEIGHTS = dict(
    out_of_bounds=0.6,
    long_gaps=0.6,
    goal=0.8,
    still_stoppage=0.3,
    player_slowdown=0.8,
    tags=0.7,
)
EVAL_CONFIG = dict(
    DEAD_THRESHOLD=-0.6,        # weighted score below this is dead
    MAX_LIVE_GAP_S=2,           # a live gap this short between two dead spells is the same stoppage
    MIN_DEAD_S=1,               # a dead spell this short between two live spells is indicator flicker
)


# ------------------------------------------------------------------------------- data ----

def load_tags():
    """Smart tagging as a DataFrame with code, start_frame, end_frame (one row per instance)."""
    root = ET.parse(require(["smart_tagging"])[0]).getroot()
    rows = [dict(code=inst.findtext('code'), start=float(inst.findtext('start')), end=float(inst.findtext('end')))
            for inst in root.iter('instance')]
    tags = pd.DataFrame(rows).sort_values('start').reset_index(drop=True)
    tags['start_frame'] = (tags['start'] * FPS).round().astype(int)
    tags['end_frame'] = (tags['end'] * FPS).round().astype(int)
    return tags


def tag_mask(tags, codes):
    """(N,) bool: True on every frame covered by a tag with one of these codes (end_frame included)."""
    mask = np.zeros(N_FRAMES, bool)
    for s, e in tags.loc[tags['code'].isin(codes), ['start_frame', 'end_frame']].to_numpy():
        mask[s:e + 1] = True
    return mask


def paint_gaps(starts, ends, value=-1):
    """(N,) int: value on every frame of the given gaps (end exclusive), 0 elsewhere."""
    out = np.zeros(N_FRAMES, dtype=int)
    for s, e in zip(starts, ends):
        out[s:e] = value
    return out


# ----------------------------------------------------------------- 1. out of bounds / gaps ----

def build_out_of_bounds(stage):
    """(N,) -1 on every ball gap Task 3 says the ball left the pitch through, 0 otherwise."""
    out = stage['gap_exit_out']
    out_of_bounds = np.zeros(N_FRAMES)
    for s, e in zip(stage['gap_start'][out], stage['gap_end'][out]):
        out_of_bounds[s:e + 1] = -1
    return out_of_bounds


def build_long_gaps(stage, tags, cfg=OUT_OF_BOUNDS_CONFIG):
    """(N,) -1 on in-bounds ball gaps with a SET PIECES tag ending near the gap's end, 0 elsewhere."""
    keep = ~stage['gap_exit_out']
    starts, ends = stage['gap_start'][keep], stage['gap_end'][keep]
    set_piece_ends = np.sort(tags.loc[tags['code'] == 'SET PIECES', 'end_frame'].to_numpy())
    lo = np.searchsorted(set_piece_ends, ends - cfg['look_ahead_seconds'] * FPS, side='left')
    hi = np.searchsorted(set_piece_ends, ends + cfg['look_beyond_seconds'] * FPS, side='right')
    hit = hi > lo
    return paint_gaps(starts[hit], ends[hit])


# -------------------------------------------------------------------------------- 2. goal ----

def near_goal_mouth(points, cfg=GOAL_CONFIG):
    """(G,) bool: point is close to either goal line and close to the centre of the goal in y."""
    length, width = PITCH
    dist_to_goal_line = np.minimum(points[:, 0], length - points[:, 0])
    dist_to_centre_y = np.abs(points[:, 1] - width / 2)
    return (dist_to_goal_line < cfg['distance_from_goal_line']) & (dist_to_centre_y < cfg['distance_from_goal_centre_y'])


def build_goal_dead(stage, tags, cfg=GOAL_CONFIG):
    """(N,) -1 on gaps where the ball was (1) last seen near a goal mouth, (2) gone long enough, (3) during a GOALS tag."""
    starts, ends = stage['gap_start'], stage['gap_end']
    last_seen = stage['ball'][np.maximum(starts - 1, 0)]          # clamped so a gap on frame 0 doesn't wrap
    goal_tag = tag_mask(tags, ['GOALS'])
    near_goal = near_goal_mouth(last_seen, cfg)
    long_enough = (ends - starts) >= cfg['gap_length_seconds'] * FPS
    overlaps_tag = np.array([goal_tag[s:e].any() for s, e in zip(starts, ends)], dtype=bool)
    is_goal_gap = near_goal & long_enough & overlaps_tag
    return paint_gaps(starts[is_goal_gap], ends[is_goal_gap])


# ------------------------------------------------------------------- 3. ball stationary ----

def compute_ball_speed(ball):
    """(N,) m/s from frame-to-frame displacement. NaN where this frame or the one before has no ball."""
    displacement = np.diff(ball, axis=0, prepend=np.full((1, 2), np.nan))
    return np.linalg.norm(displacement, axis=1) * FPS


def smooth_speed(speed, frames):
    """(N,) centred rolling median. Frames with no speed stay NaN."""
    v = pd.Series(speed).rolling(frames, center=True, min_periods=1).median().to_numpy()
    return np.where(np.isnan(speed), np.nan, v)


def stillness_counter(v, V0, gap_mode='reset'):
    """(N,) int: consecutive frames with v <= V0. A missing frame resets the count or holds it, never adds to it."""
    still_frames = np.zeros(len(v), dtype=int)
    count = 0
    for t, value in enumerate(v):
        if np.isnan(value):
            count = 0 if gap_mode == 'reset' else count
        elif value <= V0:
            count += 1
        else:
            count = 0
        still_frames[t] = count
    return still_frames


def stoppage_from_stillness(still_frames, N_still_s):
    """(N,) bool: every frame of a still spell that lasts at least N_still_s seconds."""
    stoppage = np.zeros(len(still_frames), dtype=bool)
    for is_still, start, end in runs(still_frames > 0):
        if is_still and still_frames[start:end].max() >= N_still_s * FPS:
            stoppage[start:end] = True
    return stoppage


def build_still_stoppage(stage, cfg=STILL_CONFIG):
    """(N,) -1 on every frame of a sustained low-speed spell (heavy smoothing), 0 otherwise."""
    v = smooth_speed(compute_ball_speed(stage['ball']), cfg['median_frames'])
    still_frames = stillness_counter(v, cfg['V0'], cfg['gap_mode'])
    return np.where(stoppage_from_stillness(still_frames, cfg['N_still_s']), -1, 0)


# --------------------------------------------------------------------- 4. player slowdown ----

def average_player_speed_near_ball(ball, players, player_speed, R):
    """(N,) mean speed of players within R m of the ball. NaN if nobody is near or there is no ball."""
    dist = np.linalg.norm(players - ball[:, None, :], axis=2)
    near = (dist <= R) & ~np.isnan(player_speed)
    n_near = near.sum(axis=1)
    speed_sum = np.where(near, player_speed, 0).sum(axis=1)
    return np.where(n_near > 0, speed_sum / np.maximum(n_near, 1), np.nan)


def smooth_and_slope(speed):
    """Fill NaNs, smooth with a 1 s rolling mean, and differentiate. Returns (smooth m/s, slope m/s²)."""
    filled = pd.Series(speed).interpolate(limit_direction='both')
    smooth = filled.rolling(FPS, center=True, min_periods=1).mean().to_numpy()
    return smooth, np.gradient(smooth) * FPS


def block_start(mask, i):
    """Walk back from frame i to the first frame of the True block it sits in."""
    while i > 0 and mask[i - 1]:
        i -= 1
    return i


def last_block_before(mask, start, look_back):
    """First frame of the most recent True block in the look_back frames up to `start`, or None."""
    lo = max(start - look_back, 0)
    hits = np.flatnonzero(mask[lo:start])
    return block_start(mask, lo + hits[-1]) if len(hits) else None


def find_stoppage_start(start, steep_down, missing, look_back):
    """Move a low spell's start back over the steep decline before it, then over any missing block before that."""
    stoppage_start = start
    i = last_block_before(steep_down, stoppage_start, look_back)
    if i is not None:
        stoppage_start = min(stoppage_start, i)
    i = last_block_before(missing, stoppage_start + 1, look_back + 1)    # +1: includes stoppage_start itself
    if i is not None:
        stoppage_start = min(stoppage_start, i)
    return stoppage_start


def find_stoppage_end(end, slope, steep_rise, look_ahead):
    """First frame within look_ahead after a low spell where players speed up again."""
    rising = np.flatnonzero(slope[end:end + look_ahead] > steep_rise)
    return end + rising[0] if len(rising) else end


def previous_peak(start, smooth, ball_missing, look_back, tol):
    """Walk back from `start` to the last speed peak, stopping at a missing ball or a drop of more than tol."""
    i = peak = start
    while i > max(start - look_back, 0) and not ball_missing[i - 1] and smooth[i - 1] >= smooth[peak] - tol:
        i -= 1
        if smooth[i] > smooth[peak]:
            peak = i
    return peak


def build_slowdown(stage, trk, cfg=PLAYER_SLOWDOWN_CONFIG):
    """(N,) -1 where players near the ball slow to a stop, +1 where they are clearly running, 0 otherwise."""
    slowdown = np.zeros(N_FRAMES)
    speed = average_player_speed_near_ball(stage['ball'], trk['players'], trk['player_speed'], cfg['R'])
    ball_missing = np.isnan(stage['ball'][:, 0])

    look_back, look_ahead = int(cfg['look_back_seconds'] * FPS), int(cfg['look_ahead_seconds'] * FPS)
    min_low = int(cfg['min_low_seconds'] * FPS)
    missing = np.isnan(speed)
    smooth, slope = smooth_and_slope(speed)
    steep_down = slope < cfg['steep_decline']

    # 1. every sustained low spell, stretched back over its slowdown and forward to the restart
    for is_low_spell, start, end in runs(smooth <= cfg['low']):
        if is_low_spell and end - start >= min_low:
            s = find_stoppage_start(start, steep_down, missing, look_back)
            e = find_stoppage_end(end, slope, cfg['steep_rise'], look_ahead)
            slowdown[s:e] = -1

    # 2. stretch each stoppage back to the last speed peak before it
    for is_dead, start, end in runs(slowdown == -1):
        if is_dead:
            slowdown[previous_peak(start, smooth, ball_missing, look_back, cfg['peak_tol']):start] = -1

    # 3. clearly running and not a stoppage: alive
    slowdown[(smooth > cfg['high']) & (slowdown != -1)] = 1
    return slowdown


# --------------------------------------------------------------------------------- 5. tags ----

def build_tags_stoppage(tags, cfg=TAGS_CONFIG, restarts=None):
    """(N,) -1 under stoppage tags, +1 under alive tags, 0 where neither or both (they cancel).

    With `restarts` ((N,) bool, the restart kicks), a stoppage tag stops voting dead at the first
    restart kick inside it: Smart Tagging's SET PIECES and GOALS tags cover the whole phase, the play
    after the kick included, and once the ball is kicked the stoppage is over. On our match, 67 frames
    change in the public window (spellF1 unchanged, macroF1 0.9475 -> 0.9473)."""
    stoppage = np.zeros(N_FRAMES, bool)
    for s, e in tags.loc[tags['code'].isin(cfg['stoppage_tags']), ['start_frame', 'end_frame']].to_numpy():
        end = e + 1
        if restarts is not None:
            kick = np.flatnonzero(restarts[s:e + 1])
            if len(kick):
                end = s + kick[0]
        stoppage[s:end] = True
    dead = np.where(stoppage, -1, 0)
    alive = np.where(tag_mask(tags, cfg['alive_tags']), 1, 0)
    return dead + alive


# ------------------------------------------------------------------------------- combine ----

def fill_short_gaps(mask, max_s, fill_with):
    """(N,) bool: mask, with every run of (not fill_with) of at most max_s seconds between two runs of
    fill_with set to fill_with. Runs at the very start or end of the match are left alone."""
    mask = mask.copy()
    spans = runs(mask)
    for i, (label, start, end) in enumerate(spans):
        enclosed = 0 < i < len(spans) - 1          # runs alternate, so both neighbours are fill_with
        if label != fill_with and enclosed and end - start <= max_s * FPS:
            mask[start:end] = fill_with
    return mask


def indicators():
    """{name: (N,) votes} for every indicator in WEIGHTS."""
    from pipeline.dead_refine import _restarts, ball_speed      # the restart kick dead_refine snaps to

    trk = load_tracking()
    stage = load_stage("ball_filled")
    tags = load_tags()
    restarts = _restarts(ball_speed(stage['ball'].astype(float)))
    return dict(
        out_of_bounds=build_out_of_bounds(stage),
        long_gaps=build_long_gaps(stage, tags),
        goal=build_goal_dead(stage, tags),
        still_stoppage=build_still_stoppage(stage),
        player_slowdown=build_slowdown(stage, trk),
        tags=build_tags_stoppage(tags, restarts=restarts),
    )


def evaluate_score(votes=None, weights=WEIGHTS):
    """(N,) weighted sum of the indicator votes. Negative means the indicators lean dead."""
    votes = indicators() if votes is None else votes
    names = list(weights)
    return np.array([weights[n] for n in names]) @ np.vstack([votes[n] for n in names])


def build(cfg=EVAL_CONFIG):
    """{'dead': (N,) bool, 'support': (N,) bool}. dead: weighted score below DEAD_THRESHOLD, short live
    gaps filled, then dead flickers removed. support: frames where an indicator other than
    player_slowdown votes dead (dead_refine.trim_played_starts reads it)."""
    votes = indicators()
    dead = evaluate_score(votes) < cfg['DEAD_THRESHOLD']
    dead = fill_short_gaps(dead, cfg['MAX_LIVE_GAP_S'], fill_with=True)      # join stoppages split by a brief live gap
    dead = fill_short_gaps(dead, cfg['MIN_DEAD_S'], fill_with=False)         # remove dead flickers inside live play
    support = np.any([votes[n] < 0 for n in WEIGHTS if n != 'player_slowdown'], axis=0)
    return {'dead': dead, 'support': support}


if __name__ == '__main__':
    from pipeline.common import load_public_gt
    dead = build()['dead']
    gt = load_public_gt()
    window = gt != 'X'
    truth = gt[window] == 'D'
    pred = dead[window]
    print(f"dead on {dead.mean():.1%} of frames; public window precision {truth[pred].mean():.2f}, "
          f"recall {(pred & truth).sum() / truth.sum():.2f}")
