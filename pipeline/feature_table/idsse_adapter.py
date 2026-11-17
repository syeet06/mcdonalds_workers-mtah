"""Convert Kloppy IDSSE tracking into the shared Match interface."""
import numpy as np
import pandas as pd
from kloppy.domain import Ground
from pipeline.common import HOME, AWAY
from pipeline.ball_filled import OBSERVED, MISSING
from pipeline.feature_table.build_features import Match, from_centre_origin

def idsse_to_match(tracking, match_id):
    """Return Match, frame mapping, and player mapping; no files are written."""
    metadata = tracking.metadata
    if metadata.frame_rate is None or not np.isclose(metadata.frame_rate, 25):
        raise ValueError("The shared builder requires 25 fps; resample other rates first.")

    # Use this match's actual dimensions, never the original game's defaults.
    dimensions = metadata.pitch_dimensions
    pitch = (float(dimensions.pitch_length), float(dimensions.pitch_width))
    if not np.isfinite(pitch).all() or min(pitch) <= 0:
        raise ValueError("Missing or invalid pitch dimensions")

    # Convert Kloppy records to a pandas table so columns can be selected and aligned.
    # Copying protects the loaded dataset from changes made by this adapter.
    table = tracking.to_df().copy()
    required = {"period_id", "frame_id", "ball_x", "ball_y", "ball_state"}
    if not required.issubset(table.columns):
        raise ValueError(f"Missing columns: {required - set(table.columns)}")

    # The current Match/feature code supports halves 1 and 2, not extra time.
    table = table[table["period_id"].isin([1, 2])].copy()
    if table.empty:
        raise ValueError("No regulation-half records found")
    table["frame_id"] = pd.to_numeric(table["frame_id"], errors="raise")
    if not np.equal(table["frame_id"], np.floor(table["frame_id"])).all():
        raise ValueError("Frame IDs must be integers")
    table["frame_id"] = table["frame_id"].astype(int)
    if table.duplicated(["period_id", "frame_id"]).any():
        raise ValueError("Duplicate frame IDs within a period")

    # Restore absent rows separately within each half, never across halftime.
    # Missing rows keep NaN coordinates; no interpolation is performed here.
    periods = []
    for period, group in table.groupby("period_id", sort=True):
        group = group.sort_values("frame_id").set_index("frame_id")
        group = group.reindex(range(int(group.index.min()), int(group.index.max()) + 1))
        group.index.name = "frame_id"
        group["period_id"] = period
        periods.append(group.reset_index())
    table = pd.concat(periods, ignore_index=True)
    n = len(table)

    # Metadata, rather than column order, determines each player's team.
    slots = []
    for ground, code in [(Ground.HOME, HOME), (Ground.AWAY, AWAY)]:
        teams = [team for team in metadata.teams if team.ground == ground]
        if len(teams) != 1:
            raise ValueError(f"Expected exactly one {ground} team")
        for player in teams[0].players:
            slots.append((str(player.player_id), code, player.full_name))
    if len({player_id for player_id, _, _ in slots}) != len(slots):
        raise ValueError("Duplicate player IDs in metadata")

    def coordinates(x_column, y_column):
        # Absent player columns become unknown, not (0, 0).
        xy = table.reindex(columns=[x_column, y_column]).apply(
            pd.to_numeric, errors="coerce"
        ).to_numpy(dtype=float, copy=True)     # writable: pandas copy-on-write returns read-only views
        xy[~np.isfinite(xy).all(axis=1)] = np.nan
        return xy

    ball = coordinates("ball_x", "ball_y")
    # Stack on axis 1: each player gets one stable slot across all frames.
    # The slot count may differ from the original match without changing output features.
    players = np.stack([
        coordinates(f"{player_id}_x", f"{player_id}_y")
        for player_id, _, _ in slots
    ], axis=1)
    # Slot i in this array describes the SAME player as players[:, i, :].
    team = np.array([code for _, code, _ in slots], dtype=np.int8)
    # Use IDSSE period IDs, not half_of_frame(), whose boundaries belong to our original game.
    # The shared builder uses these IDs to reset memory and prevent halftime shifts.
    half = table["period_id"].to_numpy(dtype=np.int8)

    # Check that both teams have some observed coordinates before direction inference.
    for code in (HOME, AWAY):
        if not np.isfinite(players[:, team == code]).any():
            raise ValueError(f"No tracked coordinates for team code {code}")

    # Shift both arrays together from centre-origin metres to corner-origin metres.
    ball = from_centre_origin(ball, pitch)
    players = from_centre_origin(players, pitch)

    # Provenance assumption: available provider positions are treated as observed.
    # No filled/glitch status can be established from coordinates alone.
    ball_source = np.where(np.isfinite(ball).all(axis=1), OBSERVED, MISSING).astype(np.int8)

    # Kloppy versions may export states as strings or enum objects.
    state = table["ball_state"].map(lambda value: getattr(value, "value", value))
    state = state.astype("string").str.lower()
    state_known = state.isin(["alive", "dead"])
    # False includes dead frames AND unknown states, including inserted missing rows.
    # state_known below preserves the distinction for inspection.
    live = state.eq("alive").fillna(False).to_numpy(dtype=bool)

    # Preserve provider labels for inspection only; do not assume they are audited targets.
    mapping = pd.DataFrame({
        "frame": np.arange(n), "match_id": match_id,
        "source_frame_id": table["frame_id"].to_numpy(), "half": half,
        "state_known": state_known.to_numpy(dtype=bool),
        "provider_possession_team_id": table.get(
            "ball_owning_team_id", pd.Series(np.nan, index=table.index)
        ).to_numpy(),
    })
    player_mapping = pd.DataFrame(slots, columns=["player_id", "team", "name"])
    player_mapping.index.name = "slot"

    # Package the converted arrays in the SAME interface used by load_our_match().
    # label=None means no audited target exists yet; provider labels stay out of features.
    match = Match(
        ball=ball, players=players, team=team, half=half,
        pitch=pitch, fps=25, ball_source=ball_source, live=live, label=None,
    )
    return match, mapping, player_mapping
