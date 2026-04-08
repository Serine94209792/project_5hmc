#!/usr/bin/env python3
"""
测试入口：对单个 datatype 运行 step1–step4，用可选的 pairwise（S00_vs_S10 / S00_vs_S01）
替代默认的 S00_vs_S11 来筛选差异基因并做 diffusion pseudotime，不运行 step5。

用法（在仓库根目录，与 run_pipeline 相同）:
  python -m subtyping.run_test_pipeline_steps1_4 \\
    --datatype gene_counts \\
    --matrix feature_selection/gene_counts/vst_counts.tsv \\
    --metadata cfDNA_metadata2_TNM.csv \\
    --pairwise-comparison S00_vs_S10

  # 或显式指定输出目录
  python -m subtyping.run_test_pipeline_steps1_4 \\
    --datatype gene_counts \\
    --matrix ... \\
    --output-dir subtyping/my_custom_test_dir \\
    --pairwise-comparison S00_vs_S01
"""
from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path

from subtyping.pipeline_utils import ensure_dir
from subtyping.step3_differential_analysis import DEFAULT_PAIRWISE_COMPARISON, PAIRWISE_SELECTION_CHOICES

DEFAULT_RSCRIPT = '/home/shy/anaconda3/envs/R443/bin/Rscript'
DEFAULT_PYTHON = '/home/shy/anaconda3/envs/py310/bin/python'


def run(cmd: list[str], cwd: str | Path) -> None:
    cwd_path = Path(cwd)
    print('RUN', ' '.join(cmd))
    env = os.environ.copy()
    existing = env.get('PYTHONPATH', '')
    env['PYTHONPATH'] = str(cwd_path) + (os.pathsep + existing if existing else '')
    subprocess.run(cmd, cwd=str(cwd_path), env=env, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Test subtyping steps 1–4 with S00_vs_S10 or S00_vs_S01 (or default S00_vs_S11) gene selection.'
    )
    parser.add_argument('--datatype', required=True, help='数据类型名（用于默认输出目录命名）')
    parser.add_argument('--matrix', required=True, help='原始特征矩阵路径')
    parser.add_argument('--metadata', default='cfDNA_metadata2_TNM.csv')
    parser.add_argument('--subtyping-root', default='subtyping')
    parser.add_argument('--rscript', default=DEFAULT_RSCRIPT)
    parser.add_argument('--python-bin', default=DEFAULT_PYTHON)
    parser.add_argument(
        '--pairwise-comparison',
        default='S00_vs_S10',
        choices=PAIRWISE_SELECTION_CHOICES,
        help=f'用于 step3 筛选 + HSICLasso 的比较组（默认 S00_vs_S10 便于与主流程 S00_vs_S11 对照）。可选: {PAIRWISE_SELECTION_CHOICES}',
    )
    parser.add_argument(
        '--output-dir',
        default=None,
        help='输出目录；默认 subtyping/<datatype>_test_<pairwise>（pairwise 中的 _vs_ 会写成 _vs_）',
    )
    parser.add_argument('--top-n', type=int, default=100, help='传给 step3 的 HSICLasso top_n（默认 100）')
    parser.add_argument('--step4-k', type=int, default=15)
    parser.add_argument('--step4-n-dcs', type=int, default=20)
    parser.add_argument('--step4-random-state', type=int, default=42)
    args = parser.parse_args()

    if args.top_n <= 0:
        raise ValueError('--top-n must be positive')

    repo_root = Path.cwd()
    subtyping_root = Path(args.subtyping_root)
    if args.output_dir:
        datatype_dir = ensure_dir(Path(args.output_dir))
    else:
        datatype_dir = ensure_dir(subtyping_root / f'{args.datatype}_test_{args.pairwise_comparison}')

    ensure_dir(datatype_dir / 'input')
    ensure_dir(datatype_dir / 'logs')
    ensure_dir(datatype_dir / 'tables')
    ensure_dir(datatype_dir / 'differential')
    ensure_dir(datatype_dir / 'trajectory')

    annotation_path = datatype_dir / 'input' / 'state_annotation.tsv'
    matrix_path = datatype_dir / 'matrix.tsv'
    filtered_matrix_path = datatype_dir / 'filtered_matrix.tsv'
    overlap_path = datatype_dir / 'input' / 'overlap_samples.tsv'

    run(
        [
            args.python_bin,
            str(subtyping_root / 'step1_build_state_annotation.py'),
            '--metadata',
            args.metadata,
            '--output',
            str(annotation_path),
        ],
        repo_root,
    )

    run(
        [
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
        ],
        repo_root,
    )

    run(
        [
            args.python_bin,
            str(subtyping_root / 'step3_differential_analysis.py'),
            '--matrix',
            str(matrix_path),
            '--output-dir',
            str(datatype_dir),
            '--output-matrix',
            str(filtered_matrix_path),
            '--top-n',
            str(args.top_n),
            '--pairwise-comparison',
            args.pairwise_comparison,
        ],
        repo_root,
    )

    run(
        [
            args.python_bin,
            str(subtyping_root / 'step4_diffusion_pseudotime.py'),
            '--matrix',
            str(filtered_matrix_path),
            '--output-dir',
            str(datatype_dir),
            '--k',
            str(args.step4_k),
            '--n-dcs',
            str(args.step4_n_dcs),
            '--random-state',
            str(args.step4_random_state),
        ],
        repo_root,
    )

    print()
    print('Done (steps 1–4 only). Output directory:', datatype_dir)
    print('  filtered_matrix:', filtered_matrix_path)
    print('  pseudotime:', datatype_dir / 'trajectory' / 'pseudotime.tsv')
    if args.pairwise_comparison != DEFAULT_PAIRWISE_COMPARISON:
        print(f'  pairwise used for feature selection: {args.pairwise_comparison} (not default {DEFAULT_PAIRWISE_COMPARISON})')


if __name__ == '__main__':
    main()
