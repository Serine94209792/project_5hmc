from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

STATE_ORDER = ['S00', 'S10', 'S01', 'S11']
PAIRWISE_COMPARISONS = [
    ('S00', 'S10'),
    ('S00', 'S01'),
    ('S00', 'S11'),
    ('S10', 'S01'),
    ('S10', 'S11'),
    ('S01', 'S11'),
]


def infer_sample_column(df: pd.DataFrame) -> str:
    preferred = ['sample', 'Sample', 'sample_id', 'SampleID', 'Unnamed: 0']
    for col in preferred:
        if col in df.columns:
            return col
    return df.columns[0]


def derive_batch_group(batch_value: object) -> str | None:
    if pd.isna(batch_value):
        return None
    batch = str(batch_value).upper()
    if batch.startswith('EPI'):
        return 'EPI'
    if batch.startswith('PC'):
        return 'PC'
    return None


def derive_state(lm: object, vi: object) -> str | None:
    if pd.isna(lm) or pd.isna(vi):
        return None
    lm_i = int(lm)
    vi_i = int(vi)
    mapping = {
        (0, 0): 'S00',
        (1, 0): 'S10',
        (0, 1): 'S01',
        (1, 1): 'S11',
    }
    return mapping.get((lm_i, vi_i))


def zscore_features(df: pd.DataFrame) -> pd.DataFrame:
    means = df.mean(axis=0)
    stds = df.std(axis=0, ddof=0).replace(0, 1.0)
    return (df - means) / stds


def correlation_distance(a: np.ndarray, b: np.ndarray) -> float:
    if np.allclose(a, a[0]) or np.allclose(b, b[0]):
        return 1.0
    corr = np.corrcoef(a, b)[0, 1]
    if np.isnan(corr):
        return 1.0
    return float(1.0 - corr)


def state_centroids(feature_df: pd.DataFrame, states: pd.Series) -> pd.DataFrame:
    centroids = []
    for state in STATE_ORDER:
        members = feature_df.loc[states == state]
        if members.empty:
            continue
        centroids.append(pd.Series(members.mean(axis=0), name=state))
    return pd.DataFrame(centroids)


def progression_score_method_a(feature_df: pd.DataFrame, states: pd.Series) -> pd.Series:
    root = feature_df.loc[states == 'S00']
    if root.empty:
        raise ValueError('No S00 samples found for progression score calculation.')
    centroid = root.mean(axis=0).to_numpy()
    scores = {
        sample: correlation_distance(row.to_numpy(), centroid)
        for sample, row in feature_df.iterrows()
    }
    return pd.Series(scores, name='progression_score')


def ensure_dir(path: str | Path) -> Path:
    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)
    return out
