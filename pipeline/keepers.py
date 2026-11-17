"""Goalkeeper tracks belong to the goal they stand in, not to the slot's team.

    from pipeline.keepers import keepers_by_goal
    players, team = keepers_by_goal(trk["players"], trk["team"], half_of_frame())

keepers_by_goal moves a keeper slot standing in the other team's half to a new slot of that team on
those frames. Which way each team attacks comes from the outfield slots. Public window spellF1
0.624 -> 0.650, macroF1 0.934 -> 0.941 (IDSSE has no keeper mix-ups, so there is nothing to test
there).
"""

import warnings

import numpy as np

from pipeline.ball_filled import attack_sign
from pipeline.common import AWAY, HOME, PITCH


def _home_frame_x(players, team, half, exclude=()):
    """(N, S) x in metres from the centre spot, + towards the goal home attacks; NaN outside halves.
    Directions come from every slot not in `exclude`."""
    keep = np.ones(len(team), dtype=bool)
    keep[list(exclude)] = False
    sign = attack_sign(players[:, keep], team[keep], half)[half, HOME].astype(float)
    sign[sign == 0] = np.nan
    return (players[:, :, 0].astype(float) - PITCH[0] / 2) * sign[:, None]


def keeper_slots(players, team, half):
    """(home, away) goalkeeper slots: each team's slot whose median position is deepest towards its
    own goal, among slots tracked on at least a tenth of the match's frames."""
    x = _home_frame_x(players, team, half)
    graded = half > 0
    seen = np.isfinite(x[graded]).mean(axis=0) >= 0.1
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)          # slots never tracked
        median = np.nanmedian(x[graded], axis=0)
    home = np.flatnonzero((team == HOME) & seen)
    away = np.flatnonzero((team == AWAY) & seen)
    return int(home[np.argmin(median[home])]), int(away[np.argmax(median[away])])


def keepers_by_goal(players, team, half):
    """(players, team) with two slots added, one per team. On every frame a keeper slot stands in
    the other team's half (outside the halves nothing moves), its position moves from the keeper
    slot to the new slot of the team whose goal it is at."""
    home_gk, away_gk = keeper_slots(players, team, half)
    x = _home_frame_x(players, team, half, exclude=(home_gk, away_gk))
    n = len(team)
    out = np.concatenate([players, np.full((len(players), 2, players.shape[2]), np.nan, players.dtype)], axis=1)
    team_out = np.concatenate([team, np.array([HOME, AWAY], dtype=team.dtype)])
    with np.errstate(invalid="ignore"):
        to_away = x[:, home_gk] > 0          # Barcelona's keeper slot in Athletic's half: Athletic's keeper
        to_home = x[:, away_gk] < 0          # and the other way round
    out[to_away, n + 1] = players[to_away, home_gk]
    out[to_away, home_gk] = np.nan
    out[to_home, n] = players[to_home, away_gk]
    out[to_home, away_gk] = np.nan
    return out, team_out
