from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from subtyping.pipeline_utils import derive_batch_group, derive_state


def build_annotation(metadata_path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(metadata_path)
    if 'sample' not in df.columns:
        raise KeyError('Metadata must contain a sample column.')

    ann = pd.DataFrame({
        'sample': df['sample'].astype(str),
        'LM': pd.to_numeric(df['LM'], errors='coerce'),
        'VI': pd.to_numeric(df['VascularInvasion'], errors='coerce'),
        'batch': df['batch'].astype(str),
    })
    ann['batch_group'] = df['batch'].map(derive_batch_group)
    ann['state'] = [derive_state(lm, vi) for lm, vi in zip(ann['LM'], ann['VI'])]
    ann['include_main'] = ann['state'].notna() & ann['batch_group'].notna()
    return ann


def main() -> None:
    parser = argparse.ArgumentParser(description='Build unified state annotation for subtyping pipeline.')
    parser.add_argument('--metadata', default='cfDNA_metadata2_TNM.csv')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()

    ann = build_annotation(args.metadata)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    ann.to_csv(args.output, sep='\t', index=False)
    print(f'Saved annotation: {args.output}')
    print(ann['state'].value_counts(dropna=False).to_string())


if __name__ == '__main__':
    main()
