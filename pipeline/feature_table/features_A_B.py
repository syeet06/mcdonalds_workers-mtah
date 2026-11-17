"""Calculate distance and control features without loading or saving data.

Feature outputs (one value per frame)
------------------------------------
Feature                    Output         Meaning
d_home                     metres         Nearest observed home-player distance
d_away                     metres         Nearest observed away-player distance
margin                     metres         d_away - d_home; positive means home closer
d_home_2nd                 metres         Second-nearest observed home-player distance
d_away_2nd                 metres         Second-nearest observed away-player distance
in_home_zone               0 / 1 / NaN    Home player within 1.5 m
in_away_zone               0 / 1 / NaN    Away player within 1.5 m
is_duel                    0 / 1 / NaN    Both teams within 1.5 m
is_free                    0 / 1 / NaN    Neither team within 1.5 m
run_home_control           frames         Consecutive home-only control frames
run_away_control           frames         Consecutive away-only control frames
last_control_team          +1 / -1 / 0    Home / away / no control observed yet
frames_since_last_control  frames / NaN   Age of most recent sole-control evidence

NaN means unknown. Zone flags use 1 for true and 0 for false.
Sole control requires both distances known and only one team within 1.5 m.
Control runs include the current frame and reset on duels, free balls, or
missing evidence. Team memory persists through those periods, and its age
is zero on sole-control frames. Temporal memory resets at half boundaries
and outside playing halves. Divide frame durations by FPS for seconds.
"""

import numpy as np

from pipeline.common import HOME, AWAY


def distance_features(ball, players, team):
    """Return five named arrays, each containing one value per frame.

    Inputs use load_tracking()'s format: ball (N, 2), players (N, S, 2),
    and team (S,). Coordinates and output distances are in metres.
    Missing distances are represented by NaN.
    """
    # Select all frames and both coordinates, keeping only each team's slots.
    home_players = players[:, team == HOME, :]
    away_players = players[:, team == AWAY, :]

    # Insert a player dimension into ball: (N, 2) becomes (N, 1, 2).
    # Broadcasting subtracts each frame's ball position from every player.
    home_difference = home_players - ball[:, None, :]
    away_difference = away_players - ball[:, None, :]

    # Calculate sqrt(dx**2 + dy**2) along the coordinate dimension (axis 2).
    # Each result has shape (N, number of team slots).
    home_distances = np.linalg.norm(home_difference, axis=2)
    away_distances = np.linalg.norm(away_difference, axis=2)

    # Missing positions produce NaN distances. Replace them with infinity
    # temporarily so they sort after every observed distance.
    home_valid = np.where(np.isfinite(home_distances), home_distances, np.inf)
    away_valid = np.where(np.isfinite(away_distances), away_distances, np.inf)

    # Two extra infinity columns let us select two distances even if a team
    # has fewer than two slots. They cannot displace any observed distance.
    padding = np.full((len(ball), 2), np.inf)
    home_sorted = np.sort(np.concatenate([home_valid, padding], axis=1), axis=1)
    away_sorted = np.sort(np.concatenate([away_valid, padding], axis=1), axis=1)

    # Sort across players within each frame. Column 0 is the nearest player;
    # column 1 is the second-nearest. Copies can be changed independently.
    d_home = home_sorted[:, 0].copy()
    d_away = away_sorted[:, 0].copy()
    d_home_2nd = home_sorted[:, 1].copy()
    d_away_2nd = away_sorted[:, 1].copy()

    # Restore unknown values: a missing ball makes all distances unknown;
    # fewer than two observed players makes the second distance unknown.
    for distance in (d_home, d_away, d_home_2nd, d_away_2nd):
        distance[~np.isfinite(distance)] = np.nan

    # Positive means home is closer; negative means away is closer.
    # If either nearest distance is unknown, subtraction also gives NaN.
    margin = d_away - d_home

    # Return arrays only. This function does not create a table or write files.
    return {
        "d_home": d_home,
        "d_away": d_away,
        "margin": margin,
        "d_home_2nd": d_home_2nd,
        "d_away_2nd": d_away_2nd,
    }


def control_features(d_home, d_away, half):
    """Return Feature B arrays from nearest distances and half identifiers.

    All inputs have one entry per frame. Pass half_of_frame() for a full
    match, or its matching slice when using only part of the tracking data.
    Team memory uses +1 for home, -1 for away, and 0 for no control seen yet.
    """
    d_home = np.asarray(d_home)
    d_away = np.asarray(d_away)
    half = np.asarray(half)
    if d_home.ndim != 1 or d_away.shape != d_home.shape or half.shape != d_home.shape:
        raise ValueError("Distances and half must be one-dimensional arrays of equal length")

    # A zone is known when its team's nearest distance is available.
    # Sole control, duels, and free balls require both teams' distances.
    home_known = np.isfinite(d_home)
    away_known = np.isfinite(d_away)
    known = home_known & away_known
    home_zone = d_home <= 1.5
    away_zone = d_away <= 1.5
    home_only = known & home_zone & ~away_zone
    away_only = known & away_zone & ~home_zone

    # Keep missing evidence as NaN instead of calling it a free ball.
    # np.where chooses the flag when known, and NaN otherwise.
    in_home_zone = np.where(home_known, home_zone, np.nan)
    in_away_zone = np.where(away_known, away_zone, np.nan)
    is_duel = np.where(known, home_zone & away_zone, np.nan)
    is_free = np.where(known, ~home_zone & ~away_zone, np.nan)

    n = len(d_home)
    run_home_control = np.zeros(n, dtype=int)
    run_away_control = np.zeros(n, dtype=int)
    last_control_team = np.zeros(n, dtype=np.int8)
    # Until sole control is observed, its age is unknown rather than zero.
    frames_since_last_control = np.full(n, np.nan)
    home_count = away_count = 0
    last_team = 0
    last_frame = None

    for t in range(n):
        # Reset at the first frame, each half transition, and outside halves.
        # The first condition prevents checking the previous frame at t=0.
        if t == 0 or half[t] != half[t - 1] or half[t] == 0:
            home_count = away_count = 0
            last_team = 0
            last_frame = None

        # Leave temporal outputs at their defaults outside playing halves.
        if half[t] == 0:
            continue

        # Count consecutive sole-control frames, including this frame.
        # Duels, free balls, and missing evidence break these runs.
        home_count = home_count + 1 if home_only[t] else 0
        away_count = away_count + 1 if away_only[t] else 0

        # Remember the latest sole-control team through uncertain periods.
        if home_only[t]:
            last_team = 1
            last_frame = t
        elif away_only[t]:
            last_team = -1
            last_frame = t

        run_home_control[t] = home_count
        run_away_control[t] = away_count
        last_control_team[t] = last_team
        if last_frame is not None:
            # Zero during sole control; increases on every frame after it.
            frames_since_last_control[t] = t - last_frame

    return {
        "in_home_zone": in_home_zone,
        "in_away_zone": in_away_zone,
        "is_duel": is_duel,
        "is_free": is_free,
        "run_home_control": run_home_control,
        "run_away_control": run_away_control,
        "last_control_team": last_control_team,
        "frames_since_last_control": frames_since_last_control,
    }
