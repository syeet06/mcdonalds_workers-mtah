"""Refine Task 4's dead-ball mask: set-piece anchors, ball seen over a line, snapped edges.

    from pipeline.dead_refine import refine
    dead = refine(load_stage("dead_ball")["dead"], ball, half, set_piece_starts, pitch)

Task 4's detector is precise but misses short stoppages (throw-ins, quick free kicks). Three steps,
applied in this order, each tested on 7 degraded IDSSE matches with the XGBoost possession model
(spellF1 with the real dead mask: 0.614):

dead_ball.py is Task 4's; these steps run after it and leave it untouched.
"""

import warnings

import numpy as np
import pandas as pd

from pipeline.common import AWAY, FPS, HOME, runs

# Params
MOVING_MS = 4.0          # ball at least this fast: in play (anchors)
STILL_MS = 2.0           # ball at most this fast: still (restart kick starts from here)...
KICK_MS = 5.0            # ...and reaches this on the kick
OUT_MARGIN_M = 0.5       # ball this far over a line: out
MAX_AHEAD_S = 40         # anchor: how far after the tag to look for the restart
MAX_BACK_S = 30          # anchor: how far before the tag to look for the last ball movement
CUT_NEAR_S = 2           # cut-away: ball out or a whistle counts from this long before its stoppage
VIDEO_MOVING_FRAMES = FPS // 2   # anchor: the video ball counts as play if it moves for this long on end
MAX_WAIT_S = 30          # ball out: how long to wait for the restart kick
SNAP_WINDOW = 4 * FPS    # edges move at most this far
RISE_FRAMES = 12         # snapped ends: the restart reaches KICK_MS within this many frames
MIN_DEAD_S = 2           # shorter stoppages are dropped (IDSSE truth: 0.8% are shorter)
MAX_STILL_GAP_S = 20     # a live gap this short between stoppages, ball never moving, is dead
MAX_SHORT_GAP_S = 3.5    # a live gap this short between stoppages is dead, whatever the ball does
RUN_MS = 2.0             # a player at least this fast is moving...
QUIET_PLAYERS = 3        # ...and at most this many of them on average...
QUIET_S = 3              # ...over this long: the players are quiet (public: 4% of play, 73% of stoppages)
CARRY_NEAR_M = 1.5       # carried ball: a player this close...
CARRY_MS = (1.0, 6.0)    # ...and the ball at dribbling pace (median over half a second)
CARRY_SHARE = 0.5        # a stoppage whose first 3 s are at least this much carried starts late
CARRY_CLEAR_S = 1.0      # ...where this long passes without a carry
WHISTLE_NEAR_S = 2       # a whistle this close to a stoppage's start (ball not out) times it...
WHISTLE_BACK_S = 5       # ...or this long before it (the ball rolled on after the foul)
WHISTLE_LAG = FPS // 2   # ...starting this long after the blast begins (public: 9 and 17 frames)
PLAY_SUSTAIN = FPS // 2  # played start: the ball moving at least MOVING_MS this long, inside the pitch...
CONTROL_FRAMES = 6       # ...and a player within CARRY_NEAR_M of it this many frames on end...
CONTROL_AFTER = FPS      # ...during the movement or this soon after it
PLAY_SEARCH_S = 15       # how far into a stoppage to look for that
N_REFINE_STEPS = 15       # refine()'s steps, for the progress bar
SNAP_GUARD = True        # snap_dead_edges: a start doesn't move back across a restart kick (F2)
END_AT_GO = True         # end_at_players_go: an untracked stoppage ends when the players get going (F5)
CORNER_KICK_M = 3.0      # end_at_players_go: a kick seen this close to a corner flag...
CORNER_KICK_NEAR = FPS // 2   # ...this close to a stoppage's end keeps that end (the players go on the run-up)
SEEN_RESTART_EDGE_M = 2.0     # stoppages_at_seen_restarts: the ball reappears this close inside a line (or outside)...
SEEN_RESTART_GAP_S = 2        # ...after at least this long unseen, at kick speed...
SEEN_RESTART_BACK_S = 10      # ...: dead from the gap's start, at most this long before the kick
SEEN_RESTARTS = True     # stoppages_at_seen_restarts (below)
DUEL_STOPS = True        # start_at_duel_stops: a stoppage starts where a duel stops play (below)
DUEL_M = 2.0             # duel: a home and an away player this close...
DUEL_FAST_MS = 2.5       # ...the pair running at least this fast (mean of the two) in the DUEL_PRE frames before...
DUEL_PRE = 15
DUEL_SLOW_MS = 2.0       # ...and at most this fast on average from DUEL_POST[0] to DUEL_POST[1] frames after
DUEL_POST = (15, 40)
DUEL_SEARCH_S = 5        # how far into a stoppage to look for the duel
DUEL_DROP_MS = 0.5       # everyone slows down: mean player speed falls by this much across it...
DUEL_PLAY_MS = 2.0       # ...after players ran at least this fast on average from the stoppage's start to it...
DUEL_BALL_LOST = 0.2     # ...and the tracking sees the ball at most this share of the 2 s after it


def ball_speed(ball):
    """(N,) m/s from positions, NaN where this frame or the one before has no ball."""
    return np.linalg.norm(np.diff(ball, axis=0, prepend=np.full((1, 2), np.nan)), axis=1) * FPS


def _restarts(speed):
    """(N,) bool: the ball leaves still (or reappears) on this frame and reaches KICK_MS within
    RISE_FRAMES. Unlike _kicks it catches a ball rolled away slowly and one first seen already moving."""
    prev = np.r_[np.nan, speed[:-1]]
    peak = pd.Series(speed[::-1]).rolling(RISE_FRAMES, min_periods=1).max().to_numpy()[::-1]
    with np.errstate(invalid="ignore"):
        return (speed > STILL_MS) & ((prev <= STILL_MS) | np.isnan(prev)) & (peak >= KICK_MS)


def _outside(ball, pitch, margin):
    length, width = pitch
    x, y = ball[:, 0], ball[:, 1]
    with np.errstate(invalid="ignore"):
        return (x < -margin) | (x > length + margin) | (y < -margin) | (y > width + margin)


def _video_play(ball, video, pitch):
    """(played, out): frames where the tracking has lost the ball and the video ball moves for
    VIDEO_MOVING_FRAMES on end (shorter bursts are detector jitter or a ball fetched for the
    restart), and frames where the tracking sees the ball over a line."""
    with np.errstate(invalid="ignore"):
        fast = (ball_speed(np.where(np.isnan(ball), video, ball)) >= MOVING_MS) & ~(ball_speed(ball) >= MOVING_MS)
    played = np.zeros(len(ball), dtype=bool)
    for value, s, e in runs(fast):
        if value and e - s >= VIDEO_MOVING_FRAMES:
            played[s:e] = True
    return played, _outside(ball, pitch, OUT_MARGIN_M)


def _anchor_spans(ball, half, starts, ends=None, video=None, pitch=None):
    """(start, restart) of the stoppage around each anchor frame: from the last frame before it with
    the ball moving to the restart (ball moving again after a quiet second). Anchors with no restart
    within MAX_AHEAD_S are left out. With `ends`, a restart later than its anchor's end is cut there.
    With `video` (the video ball, and `pitch`) a stoppage starts after the video ball's last
    sustained movement instead, unless the tracking saw the ball over a line before that."""
    with np.errstate(invalid="ignore"):
        moving = ball_speed(ball) >= MOVING_MS
    played, out = _video_play(ball, video, pitch) if video is not None else (None, None)
    quiet_before = pd.Series(moving.astype(float)).rolling(FPS, min_periods=1).max().shift(1).fillna(0).to_numpy() == 0
    starts = np.asarray(starts, dtype=int)
    ends = np.full(len(starts), len(half)) if ends is None else np.asarray(ends, dtype=int)
    spans = []
    for a, end in zip(starts, ends):
        if a <= 0 or a >= len(half) or half[a] == 0:
            continue
        ahead = np.flatnonzero(moving[a:a + MAX_AHEAD_S * FPS] & quiet_before[a:a + MAX_AHEAD_S * FPS])
        if not len(ahead):
            continue
        restart = min(a + ahead[0], max(end + 1, a + 1))
        lo = max(a - MAX_BACK_S * FPS, int(np.flatnonzero(half == half[a])[0]))
        back = np.flatnonzero(moving[lo:a])
        start = lo + back[-1] + 1 if len(back) else lo
        if played is not None:
            later = np.flatnonzero(played[start:a])
            if len(later) and not out[start:start + later[-1]].any():
                start += later[-1] + 1
        spans.append((start, restart))
    return spans


def anchor_set_pieces(dead, ball, half, tag_starts, tag_ends=None, video=None, pitch=None):
    """dead plus one stoppage per set-piece tag, from the last frame before the tag with the ball
    moving to the restart. Tags with no restart within MAX_AHEAD_S are skipped. With `tag_ends` (the
    tags' last frames) no stoppage runs past its tag: the restart is inside the tag. With `video`
    and `pitch` the video ball's sustained movement counts too (see _anchor_spans)."""
    out = dead.copy()
    for start, restart in _anchor_spans(ball, half, tag_starts, tag_ends, video, pitch):
        out[start:restart] = True
    return out & (half > 0)


def anchor_cutaways(dead, ball, half, pitch, starts, whistles=None, video=None):
    """dead plus one stoppage per broadcast cut-away, built like a tag's, but only where something
    else agrees play has stopped: `dead` already marks part of it, the ball is seen over a line, or
    a whistle starts, from CUT_NEAR_S before it to its restart. The TV also cuts to a replay while a
    keeper holds a saved ball and during open play, so a cut-away alone is not a stoppage."""
    outside = _outside(ball, pitch, OUT_MARGIN_M)
    whistles = np.zeros(0, dtype=int) if whistles is None else np.asarray(whistles, dtype=int)
    out = dead.copy()
    for start, restart in _anchor_spans(ball, half, starts, video=video, pitch=pitch):
        lo = max(start - CUT_NEAR_S * FPS, 0)
        if dead[start:restart].any() or outside[lo:restart].any() or ((whistles >= lo) & (whistles < restart)).any():
            out[start:restart] = True
    return out & (half > 0)


def outside_dead(dead, ball, half, pitch):
    """dead plus every stretch from the ball being seen over a line to the next restart within
    MAX_WAIT_S, or until the ball is back inside if there is none. A ball back inside already moving
    (a throw-in taken, a ball that clipped the line) ends it there."""
    outside = _outside(ball, pitch, OUT_MARGIN_M)
    speed = ball_speed(ball)
    kick = _restarts(speed)
    out = dead.copy()
    for value, s, e in runs(outside):
        if not value or half[s] == 0:
            continue
        with np.errstate(invalid="ignore"):
            back_moving = (speed[e:e + 3] >= MOVING_MS).any()
        hit = [] if back_moving else e + np.flatnonzero(kick[e:min(e + MAX_WAIT_S * FPS, len(dead))])
        out[s:hit[0] if len(hit) else e] = True
    return out & (half > 0)


def snap_dead_edges(dead, ball, half, pitch, window=SNAP_WINDOW):
    """Each stoppage starts at the first frame the ball is outside and ends at the restart kick, if
    either is within `window` frames of the edge; otherwise the edge stays. With SNAP_GUARD a start
    never moves back across a restart kick (see the loop)."""
    outside = _outside(ball, pitch, 0.0)
    kick = _restarts(ball_speed(ball))
    out = dead.copy()
    for value, s, e in runs(dead):
        if not value or half[s] == 0:
            continue
        hit = max(s - window, 0) + np.flatnonzero(outside[max(s - window, 0):min(s + window, e)])
        if len(hit):
            h = hit[0]
            if SNAP_GUARD and h + 1 < s and kick[h + 1:s].any():
                # a restart after the ball went out and before the start (a throw-in taken from behind the
                # line): play resumed, so only a fresh crossing after it counts (the ball back in, then out)
                k = h + 1 + np.flatnonzero(kick[h + 1:s])[-1]
                back = k + np.flatnonzero(~outside[k:s] & np.isfinite(ball[k:s, 0]))
                again = back[0] + np.flatnonzero(outside[back[0]:s]) if len(back) else []
                h = again[0] if len(again) else s
            out[min(s, h):max(s, h)] = h < s
        lo = max(e - window, s + 1)
        hit = lo + np.flatnonzero(kick[lo:min(e + window, len(dead))])
        if len(hit):
            out[min(e, hit[0]):max(e, hit[0])] = hit[0] > e
    return out & (half > 0)


def carried(ball, players):
    """(N,) bool: a player within CARRY_NEAR_M of the ball and the ball moving at dribbling pace."""
    dist = np.linalg.norm(players - ball[:, None, :], axis=2)
    near = np.min(np.where(np.isfinite(dist), dist, np.inf), axis=1)
    speed = pd.Series(ball_speed(ball)).rolling(FPS // 2, center=True, min_periods=1).median().to_numpy()
    with np.errstate(invalid="ignore"):
        return (near < CARRY_NEAR_M) & (speed >= CARRY_MS[0]) & (speed < CARRY_MS[1])


def controlled(ball, players, k=CONTROL_FRAMES):
    """(N,) bool: a player within CARRY_NEAR_M of the tracked ball for k frames on end."""
    with np.errstate(invalid="ignore"):
        dist = np.linalg.norm(players - ball[:, None, :], axis=2)
    near = np.min(np.where(np.isfinite(dist), dist, np.inf), axis=1) <= CARRY_NEAR_M
    out = np.zeros(len(ball), dtype=bool)
    for value, s, e in runs(near):
        if value and e - s >= k:
            out[s:e] = True
    return out


def played_ball(ball, players, pitch):
    """(N,) bool: the ball being played. A run of at least PLAY_SUSTAIN frames with the ball inside the
    pitch at MOVING_MS or more, kept only if a player gains control of it during the run or within
    CONTROL_AFTER of its end, and counted up to that last control. A ball kicked away or running loose
    after play has stopped reaches nobody, so it isn't play."""
    speed = ball_speed(ball)
    inside = ~_outside(ball, pitch, 0.0) & np.isfinite(ball[:, 0])
    with np.errstate(invalid="ignore"):
        fast = (speed >= MOVING_MS) & inside
    control = controlled(ball, players)
    out = np.zeros(len(ball), dtype=bool)
    for value, s, e in runs(fast):
        if not value or e - s < PLAY_SUSTAIN:
            continue
        got = np.flatnonzero(control[s:min(e + CONTROL_AFTER, len(ball))])
        if len(got):
            out[s:s + got[-1] + 1] = True
    return out


def trim_played_starts(dead, ball, players, pitch, support, whistles=None):
    """A stoppage can't begin while the ball is still being played. Where a stoppage's first frames
    rest on player_slowdown alone (`support`: frames where another Task 4 indicator votes dead), its
    start moves to after the last played ball (played_ball) in those frames, looking at most
    PLAY_SEARCH_S in, and never past a referee's whistle (WHISTLE_LAG after an onset from 2 s before
    the start on)."""
    played = played_ball(ball, players, pitch)
    wh = np.zeros(len(dead), dtype=bool)
    if whistles is not None and len(whistles):
        wh[np.asarray(whistles, dtype=int)] = True
    out = dead.copy()
    for value, a, e in runs(dead):
        if not value:
            continue
        lim = min(e, a + PLAY_SEARCH_S * FPS)
        backed = np.flatnonzero(support[a:lim])
        if len(backed):
            lim = a + backed[0]
        lo = max(a - 2 * FPS, 0)
        blast = np.flatnonzero(wh[lo:lim])
        if len(blast):
            lim = min(lim, max(a, lo + blast[0] + WHISTLE_LAG))
        k = np.flatnonzero(played[a:lim])
        if len(k) and a + k[-1] + 1 < e:
            out[a:a + k[-1] + 1] = False
    return out


def trim_carried_starts(dead, half, carry):
    """Stoppages that start with a player carrying the ball start where the carry ends: the first
    frame followed by CARRY_CLEAR_S with no carry (all live if it never ends)."""
    k = int(CARRY_CLEAR_S * FPS)
    clear = pd.Series((~carry).astype(float)[::-1]).rolling(k, min_periods=k).min().to_numpy()[::-1] == 1
    out = dead.copy()
    for value, s, e in runs(dead):
        if not value or half[s] == 0 or carry[s:s + 3 * FPS].mean() < CARRY_SHARE:
            continue
        hit = np.flatnonzero(clear[s:e])
        out[s:s + hit[0] if len(hit) else e] = False
    return out


def start_at_whistles(dead, ball, half, pitch, onsets):
    """A stoppage with the ball not seen out near its start (a foul or offside, not a throw-in or
    corner) starts WHISTLE_LAG after the nearest whistle onset from WHISTLE_BACK_S before its start
    (the ball rolls on after a foul) to WHISTLE_NEAR_S after it (later: a restart whistle)."""
    outside = _outside(ball, pitch, 0.0)
    onsets = np.asarray(onsets, dtype=int)
    near, back = WHISTLE_NEAR_S * FPS, WHISTLE_BACK_S * FPS
    out = dead.copy()
    for value, s, e in runs(dead):
        if not value or half[s] == 0 or outside[max(s - back, 0):min(s + near, e)].any():
            continue
        close = onsets[(onsets >= s - back) & (onsets <= s + near)]
        if not len(close):
            continue
        new = int(close[np.argmin(np.abs(close - s))]) + WHISTLE_LAG
        if new < s:
            out[new:s] = half[new:s] == half[s]
        elif new < e:
            out[s:new] = False
    return out


def stoppages_at_whistles(dead, ball, half, onsets):
    """dead plus, for each whistle onset with no stoppage within WHISTLE_NEAR_S of it, one from
    WHISTLE_LAG after it to the next restart within MAX_WAIT_S (none found: left as it is)."""
    restart = _restarts(ball_speed(ball))
    near = WHISTLE_NEAR_S * FPS
    out = dead.copy()
    for onset in np.asarray(onsets, dtype=int):
        start = onset + WHISTLE_LAG
        if start >= len(dead) or half[start] == 0 or dead[max(onset - near, 0):onset + near].any():
            continue
        hit = np.flatnonzero(restart[start + 1:min(start + MAX_WAIT_S * FPS, len(dead))])
        if len(hit):
            end = start + 1 + hit[0]
            out[start:end] = half[start:end] == half[start]
    return out


def players_moving(players):
    """(N,) mean number of players at RUN_MS or more over QUIET_S (centred); untracked: not moving."""
    step = np.linalg.norm(np.diff(players, axis=0, prepend=np.full_like(players[:1], np.nan)), axis=2) * FPS
    speed = pd.DataFrame(step).rolling(5, center=True, min_periods=1).mean().to_numpy()
    with np.errstate(invalid="ignore"):
        moving = (speed >= RUN_MS).sum(axis=1)
    return pd.Series(moving).rolling(QUIET_S * FPS, center=True, min_periods=1).mean().to_numpy()


def players_quiet(players):
    """(N,) bool: on average at most QUIET_PLAYERS players moving (players_moving)."""
    return players_moving(players) <= QUIET_PLAYERS


def hold_through_quiet(dead, ball, half, quiet):
    """A stoppage that ends while the players are still quiet (the ball moved to set up the restart)
    ends at the first restart in the QUIET_S before they get going (they react to the kick; a later
    one is a pass played on), else the last before that, else where they get going. Players still
    quiet MAX_WAIT_S later: the stoppage is left as it is."""
    restart = _restarts(ball_speed(ball))
    out = dead.copy()
    for value, s, e in runs(dead):
        if not value or e >= len(dead) or half[e] == 0 or not quiet[e]:
            continue
        go = np.flatnonzero(~quiet[e:min(e + MAX_WAIT_S * FPS + 1, len(dead))])
        if not len(go):
            continue
        go = e + go[0]
        # the kick can be on e itself, or up to half the averaging window after the averaged go
        kicks = e + np.flatnonzero(restart[e:min(go + QUIET_S * FPS // 2 + 1, len(dead))])
        near = kicks[kicks >= go - QUIET_S * FPS]
        end = near[0] if len(near) else (kicks[-1] if len(kicks) else go)
        out[e:end] = half[e:end] == half[e]
    return out


def seen_kicks(ball, pitch, min_gap=1, edge_m=None, corner_m=None):
    """(N,) bool: the tracked ball reappears on this frame after at least min_gap frames unseen and
    reaches KICK_MS within RISE_FRAMES: a kick the tracking sees from its first frame. With edge_m, only
    within that inside a line or anywhere outside the pitch (a throw-in: the ball over the thrower's head
    maps metres beyond the line); with corner_m, only within that of a corner flag."""
    length, width = pitch
    speed = np.nan_to_num(ball_speed(ball))
    peak = pd.Series(speed[::-1]).rolling(RISE_FRAMES, min_periods=1).max().to_numpy()[::-1]
    seen = np.isfinite(ball[:, 0])
    out = np.zeros(len(ball), dtype=bool)
    for value, s, _ in runs(seen):
        if not value or s < min_gap or seen[s - min_gap:s].any() or peak[s] < KICK_MS:
            continue
        x, y = ball[s]
        if edge_m is not None and min(x, length - x, y, width - y) > edge_m:   # negative: outside
            continue
        to_x, to_y = min(abs(x), abs(length - x)), min(abs(y), abs(width - y))
        if corner_m is not None and np.hypot(to_x, to_y) > corner_m:
            continue
        out[s] = True
    return out


def end_at_players_go(dead, ball, half, quiet, pitch=None):
    """A stoppage whose end the tracking can't see (the ball missing in most of its last second) ends
    where the players get going: the first frame after their last quiet one (players_quiet), if they
    stay busy from there to its end and it keeps MIN_DEAD_S. Without the ball, the players starting
    to play is the restart; the anchors otherwise wait until the ball is seen again (a corner at
    46:29: 10 s late). With `pitch`, a stoppage that already ends on a corner kick the tracking sees
    (seen_kicks within CORNER_KICK_M of a flag, CORNER_KICK_NEAR of its end) keeps that end: at a
    corner the players get going on the taker's run-up, about 1.3 s before the kick (#59 34828,
    #87 29782 were right before this step and 1.3 s early after it). The same for any radius 2-5 m
    and window 0.5-1 s; the same guard for every seen kick, not just corners, loses (a ball seen
    again after play restarted)."""
    miss = np.isnan(ball[:, 0])
    corner = np.zeros(len(dead), dtype=bool) if pitch is None else seen_kicks(ball, pitch, corner_m=CORNER_KICK_M)
    out = dead.copy()
    for value, s, e in runs(dead):
        if not value or half[s] == 0 or e >= len(dead) or miss[max(e - FPS, s):e].mean() < 0.5:
            continue
        if corner[max(e - CORNER_KICK_NEAR, s):e + CORNER_KICK_NEAR].any():
            continue                                      # the kick is seen: that is the restart
        calm = np.flatnonzero(quiet[s:e])
        if not len(calm) or quiet[e - 1]:
            continue                                      # never quiet, or still quiet at its end
        go = s + calm[-1] + 1
        if s + MIN_DEAD_S * FPS < go < e:
            out[go:e] = False
    return out


def stoppages_at_seen_restarts(dead, ball, half, pitch):
    """A restart the tracking sees, in what is still play, starts a stoppage: the ball reappears within
    SEEN_RESTART_EDGE_M inside a line, or outside the pitch, at kick speed after SEEN_RESTART_GAP_S or more unseen, and no
    stoppage covers any of that gap. The ball went out while untracked and this is the throw-in,
    corner or goal kick. Dead from the gap's start, at most SEEN_RESTART_BACK_S before the kick, to
    the kick. Three stoppages (20118, 163487, 170404), public window unchanged; the same for any line
    distance 1-3 m and gap 1-4 s. Tried and dropped: snapping the ends of existing stoppages to such
    a kick."""
    kicks = seen_kicks(ball, pitch, min_gap=SEEN_RESTART_GAP_S * FPS, edge_m=SEEN_RESTART_EDGE_M)
    seen = np.isfinite(ball[:, 0])
    out = dead.copy()
    for t in np.flatnonzero(kicks & ~dead & (half > 0)):
        g = t
        while g > 0 and not seen[g - 1]:
            g -= 1
        if dead[g:t].any() or half[g] != half[t]:
            continue
        a = max(g, t - SEEN_RESTART_BACK_S * FPS)
        out[a:t] = True
    return out


def player_speeds(players, k=5):
    """(N, S) m/s, each player's displacement over the 2k frames around a frame; NaN untracked."""
    out = np.full(players.shape[:2], np.nan)
    out[k:-k] = np.linalg.norm(players[2 * k:] - players[:-2 * k], axis=2) / (2 * k) * FPS
    return out


def _mean(x, a, b):
    """NaN-safe mean of x[a:b], NaN when empty."""
    seg = x[max(a, 0):max(b, 0)]
    seg = seg[np.isfinite(seg)]
    return seg.mean() if len(seg) else np.nan


def duel_stops(players, team, speed, lo, hi):
    """Frames in [lo, hi) where a duel stops: some home-away pair within DUEL_M, the pair's mean
    speed at least DUEL_FAST_MS somewhere in the DUEL_PRE frames before and at most DUEL_SLOW_MS on
    average DUEL_POST after. One frame per stretch of such frames, its first (a stretch already
    running at lo has none)."""
    a, b = max(lo, DUEL_PRE + 1), min(hi, len(players) - DUEL_POST[1] - 1)
    if b <= a:
        return np.zeros(0, dtype=int)
    a0 = a - 1                                             # one frame early: a stretch running at lo has no start
    home, away = np.flatnonzero(team == HOME), np.flatnonzero(team == AWAY)
    p = players[a0 - DUEL_PRE:b + DUEL_POST[1] + 1]
    v = speed[a0 - DUEL_PRE:b + DUEL_POST[1] + 1]
    off = DUEL_PRE                                         # row of frame a0 in p and v
    a = a0
    n = b - a
    hit = np.zeros(n, dtype=bool)
    with np.errstate(invalid="ignore"):
        close = np.linalg.norm(p[:, home, None] - p[:, None, away], axis=3) < DUEL_M   # (T, H, A)
        pair_v = (v[:, home, None] + v[:, None, away]) / 2
    for i in range(n):
        t = off + i
        c = close[t]
        if not c.any():
            continue
        with np.errstate(invalid="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            before = np.nanmax(pair_v[t - DUEL_PRE:t + 1], axis=0)
            after = np.nanmean(pair_v[t + DUEL_POST[0]:t + DUEL_POST[1] + 1], axis=0)
        hit[i] = (c & (before >= DUEL_FAST_MS) & (after <= DUEL_SLOW_MS)).any()
    starts = np.flatnonzero(hit & ~np.r_[True, hit[:-1]]) + a      # first frames of stretches after a0
    return starts[starts >= lo]


def everyone_speed(players):
    """(N,) mean speed of all tracked players (player_speeds), m/s; NaN with nobody tracked."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmean(player_speeds(players), axis=1)


def slowdown(everyone):
    """(N,) how much everyone slows down across each frame: mean speed over the 2 s before it minus
    the mean over 0.5-3 s after it (NaN-safe), m/s."""
    n = len(everyone)
    ok = np.isfinite(everyone)
    c = np.r_[0.0, np.cumsum(np.where(ok, everyone, 0.0))]
    k = np.r_[0, np.cumsum(ok)]
    t = np.arange(n)

    def mean(a, b):
        a, b = np.clip(t + a, 0, n), np.clip(t + b, 0, n)
        with np.errstate(invalid="ignore", divide="ignore"):
            return (c[b] - c[a]) / (k[b] - k[a])
    return mean(-2 * FPS, 0) - mean(FPS // 2, 3 * FPS)


def duel_frames(players, team, chunk=5000):
    """(N,) bool: the frames duel_stops marks, over the whole match (for the viewer)."""
    out = np.zeros(len(players), dtype=bool)
    speed = player_speeds(players)
    for lo in range(0, len(players), chunk):
        out[duel_stops(players, team, speed, lo, lo + chunk)] = True
    return out


def start_at_duel_stops(dead, ball, half, players, team):
    """A stoppage whose start comes too early, while play goes on, starts where a duel stops play: in
    its first DUEL_SEARCH_S, a home and an away player running together slow to a walk
    (duel_stops), everyone slows down across that moment (slowdown at least DUEL_DROP_MS), the
    players were still running from the stoppage's start to it (DUEL_PLAY_MS) and the tracking
    loses the ball right after it (at most DUEL_BALL_LOST of the next 2 s seen). A foul: the
    challenge ends, the whistle stops everyone, the camera leaves the ball. Of several, the one with
    the biggest slowdown; the stoppage keeps MIN_DEAD_S. Starts only move later. 8 starts
    move (up to 4.4 s), public window unchanged."""
    speed = player_speeds(players)
    everyone = everyone_speed(players)
    drop = slowdown(everyone)
    seen = np.isfinite(ball[:, 0])
    out = dead.copy()
    for value, s, e in runs(dead):
        if not value or half[s] == 0:
            continue
        hi = min(s + DUEL_SEARCH_S * FPS, e - MIN_DEAD_S * FPS)
        best = None
        for t in duel_stops(players, team, speed, s + 1, hi):
            if drop[t] >= DUEL_DROP_MS and _mean(everyone, s, t) >= DUEL_PLAY_MS \
                    and seen[t:t + 2 * FPS].mean() <= DUEL_BALL_LOST and (best is None or drop[t] > drop[best]):
                best = t
        if best is not None:
            out[s:best] = False
    return out


def bridge_still_gaps(dead, ball, half, max_gap_s=MAX_STILL_GAP_S):
    """dead plus every live gap of at most max_gap_s between two stoppages of one half in which the
    tracked ball never reaches MOVING_MS: nothing was played, so it is one stoppage. Stoppages
    drop_short will remove are dropped first, so they cannot anchor a gap."""
    with np.errstate(invalid="ignore"):
        moving = ball_speed(ball) >= MOVING_MS
    out = drop_short(dead, half)                          # slivers about to go cannot anchor a gap
    r = list(runs(out))
    for (before, _, _), (value, s, e), (after, _, _) in zip(r, r[1:], r[2:]):
        if before and after and not value and half[s] > 0 and half[s] == half[e - 1] \
                and e - s <= max_gap_s * FPS and not moving[s + 1:e].any():
            out[s:e] = True
    return out


def bridge_short_gaps(dead, half, max_gap_s=MAX_SHORT_GAP_S):
    """dead plus every live gap of at most max_gap_s between two stoppages of one half, whatever the
    ball does: too short to be a possession between them. Stoppages drop_short will remove are
    dropped first, so they cannot anchor a gap."""
    out = drop_short(dead, half)
    r = list(runs(out))
    for (before, _, _), (value, s, e), (after, _, _) in zip(r, r[1:], r[2:]):
        if before and after and not value and half[s] > 0 and half[s] == half[e - 1] and e - s <= max_gap_s * FPS:
            out[s:e] = True
    return out


def drop_short(dead, half, min_s=MIN_DEAD_S):
    """dead without the stoppages shorter than min_s seconds: each one splits a possession in two."""
    out = dead.copy()
    for value, s, e in runs(dead):
        if value and half[s] > 0 and e - s < min_s * FPS:
            out[s:e] = False
    return out


def refine(dead, ball, half, tag_starts, pitch, players=None, whistles=None, tag_ends=None, cutaways=None,
           video=None, trace=None, support=None, team=None):
    """Task 4's mask with the steps above, in the order they were tested. Without `team` ((S,) per
    player slot) the duel-stop starts are skipped. Without players or
    `support` (Task 4's stage) the played-start trim is skipped, without players the
    carried-start trim is skipped, without whistle onsets the whistle timing, without tag ends the
    cap on tag stoppages, without cut-aways their stoppages, without the video ball the anchors'
    stoppages start where the tracking last saw the ball move. With a list as `trace`, (step name,
    mask after it) is appended for every step, skipped ones included (mask unchanged); the viewer
    shows them."""
    from pipeline import progress
    bar = progress.Bar(N_REFINE_STEPS, "refinement steps")
    done = [0]

    def step(name, mask):
        if trace is not None:
            trace.append((name, mask.copy()))
        done[0] += 1
        bar.update(done[0])
        if done[0] == N_REFINE_STEPS:
            bar.close()
        return mask

    dead = step("trim_played_starts", dead if players is None or support is None else
                trim_played_starts(dead, ball, players, pitch, support, whistles))
    dead = step("anchor_set_pieces", anchor_set_pieces(dead, ball, half, tag_starts, tag_ends, video, pitch))
    dead = step("anchor_cutaways", dead if cutaways is None else
                anchor_cutaways(dead, ball, half, pitch, cutaways, whistles, video))
    dead = step("outside_dead", outside_dead(dead, ball, half, pitch))
    dead = step("snap_dead_edges", snap_dead_edges(dead, ball, half, pitch))
    dead = step("trim_carried_starts", dead if players is None else trim_carried_starts(dead, half, carried(ball, players)))
    dead = step("start_at_whistles", dead if whistles is None else start_at_whistles(dead, ball, half, pitch, whistles))
    dead = step("stoppages_at_whistles", dead if whistles is None else stoppages_at_whistles(dead, ball, half, whistles))
    dead = step("hold_through_quiet", dead if players is None else hold_through_quiet(dead, ball, half, players_quiet(players)))
    dead = step("end_at_players_go", dead if players is None or not END_AT_GO else
                end_at_players_go(dead, ball, half, players_quiet(players), pitch))
    dead = step("bridge_still_gaps", bridge_still_gaps(dead, ball, half))
    dead = step("bridge_short_gaps", bridge_short_gaps(dead, half))
    dead = step("start_at_duel_stops", dead if players is None or team is None or not DUEL_STOPS else
                start_at_duel_stops(dead, ball, half, players, team))
    dead = step("stoppages_at_seen_restarts", stoppages_at_seen_restarts(dead, ball, half, pitch) if SEEN_RESTARTS else dead)
    return step("drop_short", drop_short(dead, half))
