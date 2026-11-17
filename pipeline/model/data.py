"""Load the IDSSE training tables (clean or degraded) as model-ready X, y, split by match.

    from pipeline.model.data import load, MATCHES

    clean = load("clean")
    train, test = clean.split(train=MATCHES[:2], test=MATCHES[2:3])
    train.X, train.y            # 68 features in summary order, labels 1 = home, 0 = away

Only labelled live frames are kept. NaNs are left in place: XGBoost handles them natively, and
in the degraded table the gaps are the point. Never split by frame -- neighbouring frames are
near-duplicates, so a random split leaks. Split by match_id with Dataset.split().

Build the tables first if they are missing (from the repository root):
    python3 -m pipeline.feature_table.build_idsse_features
    python3 -m pipeline.feature_table.build_idsse_features --degraded
"""
import dataclasses
import json
from pathlib import Path

import numpy as np
import pandas as pd

FEATURE_DIR = Path(__file__).resolve().parents[1] / "feature_table"
VERSIONS = {
    "clean": ("idsse_all_matches_features.csv.gz", "idsse_all_matches_summary.json"),
    "degraded": ("idsse_all_matches_features_degraded.csv.gz", "idsse_all_matches_summary_degraded.json"),
}
ID_COLUMNS = ["match_id", "half", "frame"]
LABEL = "label"


def _paths(version):
    if version not in VERSIONS:
        raise ValueError(f"version must be one of {sorted(VERSIONS)}, got {version!r}")
    table, summary = VERSIONS[version]
    return FEATURE_DIR / table, FEATURE_DIR / summary


def read_summary(version="clean"):
    return json.loads(_paths(version)[1].read_text())


def feature_columns(version="clean"):
    """The 68 model inputs, in the order every model must see them."""
    return list(read_summary(version)["matches"][0]["feature_columns"])


# Match ids in the order the builder wrote them.
MATCHES = [m["match_id"] for m in read_summary("clean")["matches"]]


@dataclasses.dataclass
class Dataset:
    """Labelled frames from one table. X, y and meta share one row index."""
    version: str
    X: pd.DataFrame          # float32 features, NaN where unobserved
    y: pd.Series             # int8, 1 = home, 0 = away
    meta: pd.DataFrame       # match_id, half, frame

    @property
    def matches(self):
        return list(pd.unique(self.meta["match_id"]))

    def __len__(self):
        return len(self.y)

    def select(self, matches):
        """The rows of these matches only."""
        matches = [matches] if isinstance(matches, str) else list(matches)
        unknown = set(matches) - set(self.matches)
        if unknown:
            raise KeyError(f"matches not in this table: {sorted(unknown)}")
        keep = self.meta["match_id"].isin(matches).to_numpy()
        return Dataset(self.version, self.X[keep], self.y[keep], self.meta[keep])

    def split(self, train, test):
        """(train, test) by match id; refuses to put a match on both sides."""
        overlap = set([train] if isinstance(train, str) else train) & \
                  set([test] if isinstance(test, str) else test)
        if overlap:
            raise ValueError(f"matches on both sides of the split: {sorted(overlap)}")
        return self.select(train), self.select(test)


def load(version="clean", matches=None):
    """Labelled frames of the clean or degraded IDSSE table, optionally only some matches."""
    table_path, _ = _paths(version)
    if not table_path.exists():
        flag = " --degraded" if version == "degraded" else ""
        raise FileNotFoundError(f"{table_path.name} not found; build it with "
                                f"python3 -m pipeline.feature_table.build_idsse_features{flag}")
    features = feature_columns(version)
    dtypes = {c: np.float32 for c in features}
    table = pd.read_csv(table_path, usecols=ID_COLUMNS + features + [LABEL], dtype=dtypes)
    table = table[table[LABEL].isin([0, 1])].reset_index(drop=True)
    data = Dataset(version, table[features], table[LABEL].astype(np.int8), table[ID_COLUMNS])
    return data if matches is None else data.select(matches)


def check_aligned(a, b):
    """Raise unless two Datasets have the same frames and labels in the same order.

    Clean and degraded tables should: degrade.py only blanks positions, never rows or labels.
    """
    if len(a) != len(b):
        raise AssertionError(f"different row counts: {len(a)} vs {len(b)}")
    if not a.meta.reset_index(drop=True).equals(b.meta.reset_index(drop=True)):
        raise AssertionError("match_id/half/frame differ between the tables")
    if not np.array_equal(a.y.to_numpy(), b.y.to_numpy()):
        raise AssertionError("labels differ between the tables")
