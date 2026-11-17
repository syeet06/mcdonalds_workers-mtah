"""Build all seven IDSSE matches with the shared builder.

Run from the repository root: python3 -m pipeline.feature_table.build_idsse_features
Output is a compressed CSV, readable with pandas.read_csv(). Raw downloads and
per-match checkpoints are cached; no original competition data is exported.

Add --degraded for the augmented set: ball and player positions blanked to look like our ATD
(degrade.py), same rows and labels as the clean set, written to *_degraded.csv.gz.
"""
import argparse
import gc
import gzip
import json
import shutil
import ssl
import urllib.request
from collections import Counter
from pathlib import Path

import certifi
import numpy as np
import pandas as pd
from lxml import etree
from kloppy import sportec

from pipeline.feature_table.idsse_adapter import idsse_to_match
from pipeline.feature_table.build_features import build_feature_table, load_our_match
from pipeline.feature_table.degrade import degradation_stats, degrade_match, seed_for

MATCH_IDS = ('J03WMX', 'J03WN1', 'J03WPY', 'J03WOH', 'J03WQQ', 'J03WOY', 'J03WR9')
HERE = Path(__file__).resolve().parent
CACHE = HERE.parent / 'cache' / 'idsse'
HF_URL = 'https://huggingface.co/datasets/pysport/idsse-data/resolve/main/'
OUTPUT = HERE / 'idsse_all_matches_features.csv.gz'
STATS_COLUMNS = ['frame', 'half', 'ball_source', 'is_duel', 'n_players_tracked', 'centroid_x_diff']


def our_match_stats():
    """Degradation targets measured on our own match, for the summary; None without its data."""
    try:
        return degradation_stats(build_feature_table(load_our_match()))
    except Exception as error:                       # teammates without data/ still get a build
        print(f'Our match stats skipped: {error}', flush=True)
        return None


def download(match_id, kind):
    """Keep reusable source XML locally; replace partial downloads only on success.

    Files come from the Hugging Face mirror: figshare (kloppy's get_IDSSE_url) answers scripts
    with an empty 202 bot challenge. certifi supplies the certificates python.org builds lack.
    """
    competition = '000001' if match_id in ('J03WMX', 'J03WN1') else '000002'
    prefix = {'meta': 'DFL_02_01_matchinformation', 'tracking': 'DFL_04_03_positions_raw_observed'}[kind]
    name = f'{prefix}_DFL-COM-{competition}_DFL-MAT-{match_id}.xml'
    path = CACHE / name
    if not path.exists():
        print(f'Downloading {match_id} {kind}', flush=True)
        temporary = path.with_suffix('.download')
        context = ssl.create_default_context(cafile=certifi.where())
        with urllib.request.urlopen(HF_URL + name, timeout=180, context=context) as response, \
                temporary.open('wb') as output:
            if response.status != 200:
                raise RuntimeError(f'{name}: HTTP {response.status}')
            shutil.copyfileobj(response, output)
        if temporary.stat().st_size == 0:
            temporary.unlink()
            raise RuntimeError(f'{name}: empty download')
        temporary.replace(path)
    return path


def raw_ball_labels(path):
    """Read independent raw codes, avoiding the loader's fallback-to-away mapping.

    Sportec BallPossession: 1 home, 2 away; other codes remain unknown.
    BallStatus: 1 live, 0 dead. Keep codes for auditing in the export.
    """
    rows = []
    period = None
    is_ball = False
    periods = {'firstHalf': 1, 'secondHalf': 2}
    for event, element in etree.iterparse(str(path), events=('start', 'end'), huge_tree=True):
        if event == 'start' and element.tag == 'FrameSet':
            is_ball = element.get('TeamId') == 'BALL'
            period = periods.get(element.get('GameSection'))
        elif event == 'end' and element.tag == 'Frame':
            if is_ball and period is not None:
                rows.append((period, int(element.get('N')), element.get('BallPossession'), element.get('BallStatus')))
            element.clear()
            while element.getprevious() is not None:
                del element.getparent()[0]
        elif event == 'end' and element.tag == 'FrameSet':
            element.clear()
    labels = pd.DataFrame(rows, columns=['half', 'source_frame_id', 'raw_possession', 'raw_ball_status'])
    if labels.duplicated(['half', 'source_frame_id']).any():
        raise ValueError('Duplicate raw ball frame IDs')
    return labels


def build_all(degraded=False):
    CACHE.mkdir(parents=True, exist_ok=True)
    suffix = '_degraded' if degraded else ''
    output = HERE / f'idsse_all_matches_features{suffix}.csv.gz'
    reports = []
    schema = None
    checkpoints = []
    for match_id in MATCH_IDS:
        checkpoint = CACHE / f'{match_id}_features{suffix}.csv.gz'
        report_path = CACHE / f'{match_id}_report{suffix}.json'
        if checkpoint.exists() and report_path.exists():
            report = json.loads(report_path.read_text())
            columns = pd.read_csv(checkpoint, nrows=0).columns.tolist()
            if 'degradation_stats' not in report:     # checkpoints from before degrade.py
                report['degradation_stats'] = degradation_stats(pd.read_csv(checkpoint, usecols=STATS_COLUMNS))
                report_path.write_text(json.dumps(report, indent=2)+'\n')
            print(f'Reusing {match_id} checkpoint: {report["rows"]:,} rows', flush=True)
        else:
            raw = download(match_id, 'tracking')
            meta = download(match_id, 'meta')
            labels = raw_ball_labels(raw)
            print(f'{match_id} raw codes: {dict(Counter(labels.raw_possession))}', flush=True)
            tracking = sportec.load_tracking(raw_data=str(raw), meta_data=str(meta), coordinates='sportec', only_alive=False)
            match, mapping, slots = idsse_to_match(tracking, match_id)
            # Free Kloppy frame objects before calculating the full feature matrix.
            del tracking
            gc.collect()
            if degraded:
                # Blank positions only; rows, halves, live mask and labels stay as in the clean set.
                match = degrade_match(match, seed_for(match_id))
            table = build_feature_table(match, drop_dead=False)
            table = table.merge(mapping[['frame', 'match_id', 'source_frame_id']], on='frame', validate='one_to_one')
            table = table.merge(labels, on=['half', 'source_frame_id'], how='left', validate='one_to_one')
            # Labels are taken from explicit raw codes only; never from proximity features.
            # Keep live rows with unknown targets for inspection; exclude NaN labels in training.
            table['label'] = table['raw_possession'].map({'1': 1.0, '2': 0.0})
            table = table.loc[table['raw_ball_status'].eq('1')].reset_index(drop=True)
            if table.empty:
                raise ValueError(f'{match_id}: no live records')
            feature_columns = [c for c in table if c not in {'frame','half','match_id','source_frame_id','raw_possession','raw_ball_status','label'}]
            if len(feature_columns) != 68:
                raise ValueError(f'Unexpected feature count: {len(feature_columns)}')
            tracked = np.isfinite(match.players).all(axis=2)
            report = {'match_id': match_id, 'rows': len(table), 'labelled_rows': int(table.label.notna().sum()),
                      'home_rows': int(table.label.eq(1).sum()), 'away_rows': int(table.label.eq(0).sum()),
                      'raw_possession_codes': {str(k): int(v) for k,v in Counter(labels.raw_possession).items()},
                      'feature_columns': feature_columns, 'pitch': list(match.pitch),
                      'max_observed_home': int(tracked[:, match.team == 0].sum(axis=1).max()),
                      'max_observed_away': int(tracked[:, match.team == 1].sum(axis=1).max()),
                      'degraded': degraded, 'seed': seed_for(match_id) if degraded else None,
                      'degradation_stats': degradation_stats(table)}
            columns = table.columns.tolist()
            temporary = checkpoint.with_suffix('.tmp')
            table.to_csv(temporary, index=False, compression='gzip')
            temporary.replace(checkpoint)
            report_path.write_text(json.dumps(report, indent=2)+'\n')
            print(f'Built {match_id}: {len(table):,} live rows; {report["labelled_rows"]:,} labelled', flush=True)
            del table, match, mapping, slots, labels, tracked
            gc.collect()
        if schema is None:
            schema = columns
        elif columns != schema:
            raise ValueError(f'{match_id}: mismatched column schema')
        checkpoints.append(checkpoint)
        reports.append(report)

    # Stream checkpoints into one gzip file, keeping just the first header.
    temporary = output.with_suffix('.tmp')
    with gzip.open(temporary, 'wt') as combined:
        for index, checkpoint in enumerate(checkpoints):
            with gzip.open(checkpoint, 'rt') as source:
                header = next(source)
                if index == 0:
                    combined.write(header)
                shutil.copyfileobj(source, combined)
    temporary.replace(output)
    if degraded:
        ball_note = ('Degraded (degrade.py): ball and player positions blanked to match our ATD; '
                     'same rows and labels as the clean set.')
    else:
        ball_note = 'Present provider ball positions treated as observed; no extra imputation.'
    summary = {'source': 'https://huggingface.co/datasets/pysport/idsse-data', 'license': 'CC-BY-4.0',
               'output': output.name, 'degraded': degraded,
               'total_rows': sum(r['rows'] for r in reports), 'matches': reports,
               'degradation_stats_all': degradation_stats(pd.concat(
                   [pd.read_csv(c, usecols=STATS_COLUMNS + ['match_id']) for c in checkpoints], ignore_index=True)),
               'degradation_stats_our_match': our_match_stats(),
               'notes': ['25 fps; regulation halves; live frames only.',
                         'Labels: raw BallPossession 1=home, 2=away; all others unknown.',
                         ball_note,
                         'Split by match_id; exclude identifiers and raw codes from X.',
                         'Includes future-frame features: offline prediction only.']}
    (HERE / f'idsse_all_matches_summary{suffix}.json').write_text(json.dumps(summary, indent=2)+'\n')
    print(f'SAVED {output}: {summary["total_rows"]:,} rows across {len(reports)} matches', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--degraded', action='store_true', help='build the augmented (degraded) set')
    build_all(degraded=parser.parse_args().degraded)
