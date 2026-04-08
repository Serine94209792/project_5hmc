from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path

from subtyping.pipeline_utils import ensure_dir

DEFAULT_RSCRIPT = '/home/shy/anaconda3/envs/R443/bin/Rscript'
DEFAULT_PYTHON = '/home/shy/anaconda3/envs/py310/bin/python'


def run(cmd: list[str], cwd: str | Path) -> None:
    """
    Execute a subprocess within the repo root while ensuring the package
    import path (`subtyping.*`) is resolvable for step scripts.
    """
    cwd_path = Path(cwd)
    print('RUN', ' '.join(cmd))
    env = os.environ.copy()
    # Make `import subtyping...` work even when the child step script is executed
    # via a filesystem path (so sys.path[0] becomes the subtyping/ directory).
    existing = env.get('PYTHONPATH', '')
    env['PYTHONPATH'] = str(cwd_path) + (os.pathsep + existing if existing else '')
    subprocess.run(cmd, cwd=str(cwd_path), env=env, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description='Run subtyping workflow for one datatype.')
    parser.add_argument('--datatype', required=True)
    parser.add_argument('--matrix', required=True)
    parser.add_argument('--metadata', default='cfDNA_metadata2_TNM.csv')
    parser.add_argument('--subtyping-root', default='subtyping')
    parser.add_argument('--rscript', default=DEFAULT_RSCRIPT)
    parser.add_argument('--python-bin', default=DEFAULT_PYTHON)
    parser.add_argument('--step5-fdr-threshold', type=float, default=0.05)
    parser.add_argument('--step5-min-abs-rho', type=float, default=0.0)
    args = parser.parse_args()

    repo_root = Path.cwd()
    subtyping_root = Path(args.subtyping_root)
    datatype_dir = ensure_dir(subtyping_root / args.datatype)
    ensure_dir(datatype_dir / 'input')
    ensure_dir(datatype_dir / 'logs')
    ensure_dir(datatype_dir / 'tables')
    ensure_dir(datatype_dir / 'differential')
    ensure_dir(datatype_dir / 'trajectory')

    annotation_path = datatype_dir / 'input' / 'state_annotation.tsv'
    matrix_path = datatype_dir / 'matrix.tsv'
    filtered_matrix_path = datatype_dir / 'filtered_matrix.tsv'
    overlap_path = datatype_dir / 'input' / 'overlap_samples.tsv'

    run([
        args.python_bin,
        str(subtyping_root / 'step1_build_state_annotation.py'),
        '--metadata',
        args.metadata,
        '--output',
        str(annotation_path),
    ], repo_root)

    run([
        args.rscript,
        str(subtyping_root / 'step2_preprocess_datatype.R'),
        '--input_matrix',
        args.matrix,
        '--annotation',
        str(annotation_path),
        '--output_matrix',
        str(matrix_path),
        '--output_overlap',
        str(overlap_path),
    ], repo_root)

    run([
        args.python_bin,
        str(subtyping_root / 'step3_differential_analysis.py'),
        '--matrix',
        str(matrix_path),
        '--output-dir',
        str(datatype_dir),
        '--output-matrix',
        str(filtered_matrix_path),
        '--top-n',
        '100',
    ], repo_root)

    run([
        args.python_bin,
        str(subtyping_root / 'step4_diffusion_pseudotime.py'),
        '--matrix',
        str(filtered_matrix_path),
        '--output-dir',
        str(datatype_dir),
    ], repo_root)

    pseudotime_path = datatype_dir / 'trajectory' / 'pseudotime.tsv'

    run([
        args.python_bin,
        str(subtyping_root / 'step5_psudotime_gene.py'),
        '--matrix',
        str(filtered_matrix_path),
        '--full-matrix',
        str(matrix_path),
        '--pseudotime',
        str(pseudotime_path),
        '--output-dir',
        str(datatype_dir),
        '--fdr-threshold',
        str(args.step5_fdr_threshold),
        '--min-abs-rho',
        str(args.step5_min_abs_rho),
    ], repo_root)


if __name__ == '__main__':
    main()
