from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import kruskal, mannwhitneyu

from subtyping.pipeline_utils import PAIRWISE_COMPARISONS, STATE_ORDER, ensure_dir, infer_sample_column

SIGNIFICANCE_THRESHOLD = 0.05
DEFAULT_HSIC_TOP_N = 100

# 用于差异基因筛选 + HSICLasso 的 pairwise 键（须与 run_pairwise 输出的名称一致，如 S00_vs_S11）
DEFAULT_PAIRWISE_COMPARISON = 'S00_vs_S11'
PAIRWISE_SELECTION_CHOICES = ('S00_vs_S11', 'S00_vs_S10', 'S00_vs_S01')


def parse_pairwise_key(pairwise_key: str) -> tuple[str, str]:
    if '_vs_' not in pairwise_key:
        raise ValueError(f'Invalid pairwise key (expected *_*_vs_*_*): {pairwise_key!r}')
    a, b = pairwise_key.split('_vs_', 1)
    if a not in STATE_ORDER or b not in STATE_ORDER:
        raise ValueError(f'Unknown state in pairwise key {pairwise_key!r}; expected states in {STATE_ORDER}')
    return a, b


def plot_volcano(df: pd.DataFrame, output_path: Path, title: str) -> None:
    if df.empty:
        return

    plot_df = df.copy()
    plot_df['neg_log10_pvalue'] = -np.log10(plot_df['pvalue'].clip(lower=1e-300))

    significant = plot_df['pvalue'] < SIGNIFICANCE_THRESHOLD

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(
        plot_df.loc[~significant, 'delta_mean'],
        plot_df.loc[~significant, 'neg_log10_pvalue'],
        s=10,
        alpha=0.5,
        color='#9ca3af',
        label='p>=0.05',
    )
    ax.scatter(
        plot_df.loc[significant, 'delta_mean'],
        plot_df.loc[significant, 'neg_log10_pvalue'],
        s=12,
        alpha=0.8,
        color='#dc2626',
        label='p<0.05',
    )
    ax.axhline(-np.log10(SIGNIFICANCE_THRESHOLD), color='#2563eb', linestyle='--', linewidth=1)
    ax.set_xlabel('delta_mean')
    ax.set_ylabel('-log10(pvalue)')
    ax.set_title(title)
    ax.legend(loc='upper right', frameon=False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def benjamini_hochberg(pvals: pd.Series) -> pd.Series:
    p = pvals.fillna(1.0).to_numpy(dtype=float)
    n = len(p)
    order = np.argsort(p)
    ranked = p[order]
    q = np.empty(n, dtype=float)
    prev = 1.0
    for i in range(n - 1, -1, -1):
        rank = i + 1
        val = min(prev, ranked[i] * n / rank)
        q[i] = val
        prev = val
    out = np.empty(n, dtype=float)
    out[order] = q
    return pd.Series(out, index=pvals.index)


def load_matrix(path: str | Path) -> tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(path, sep='\t')
    sample_col = infer_sample_column(df)
    df = df.rename(columns={sample_col: 'sample'})
    if 'state' not in df.columns:
        raise KeyError('Matrix must contain a state column.')
    df['sample'] = df['sample'].astype(str)
    df['state'] = df['state'].astype(str)
    feature_df = df.drop(columns=['sample', 'state']).apply(pd.to_numeric, errors='coerce').fillna(0.0)
    feature_df.index = df['sample']
    states = df.set_index('sample')['state']
    return feature_df, states


def _build_state_matrices(feature_df: pd.DataFrame, states: pd.Series) -> dict[str, np.ndarray]:
    return {
        state: feature_df.loc[states == state].to_numpy(dtype=float, copy=False)
        for state in STATE_ORDER
        if np.any(states.to_numpy() == state)
    }


def run_global_differential(feature_df: pd.DataFrame, states: pd.Series) -> pd.DataFrame:
    feature_names = feature_df.columns.to_numpy(dtype=str)
    state_matrices = _build_state_matrices(feature_df, states)
    groups = [state_matrices[state] for state in STATE_ORDER if state in state_matrices and state_matrices[state].shape[0] > 0]
    if len(groups) < 2:
        columns = ['feature', 'statistic', 'pvalue', 'significant_p', *[f'mean_{k}' for k in STATE_ORDER], 'fdr']
        return pd.DataFrame(columns=columns)

    stat, pvalue = kruskal(*groups, axis=0, nan_policy='omit')
    grouped_means = feature_df.groupby(states).mean().reindex(STATE_ORDER)

    out = pd.DataFrame(
        {
            'feature': feature_names,
            'statistic': np.asarray(stat, dtype=float),
            'pvalue': np.asarray(pvalue, dtype=float),
            'significant_p': np.asarray(pvalue, dtype=float) < SIGNIFICANCE_THRESHOLD,
        }
    )
    for state in STATE_ORDER:
        if state in grouped_means.index:
            out[f'mean_{state}'] = grouped_means.loc[state].to_numpy(dtype=float)
        else:
            out[f'mean_{state}'] = np.nan
    out['fdr'] = benjamini_hochberg(out['pvalue'])
    return out.sort_values(['pvalue', 'feature'])


def run_pairwise(feature_df: pd.DataFrame, states: pd.Series) -> dict[str, pd.DataFrame]:
    feature_names = feature_df.columns.to_numpy(dtype=str)
    state_matrices = _build_state_matrices(feature_df, states)
    results = {}
    for a, b in PAIRWISE_COMPARISONS:
        x = state_matrices.get(a)
        y = state_matrices.get(b)
        if x is None or y is None or x.shape[0] == 0 or y.shape[0] == 0:
            df = pd.DataFrame(columns=['feature', 'group_a', 'group_b', 'statistic', 'pvalue', 'significant_p', 'mean_a', 'mean_b', 'delta_mean', 'fdr'])
        else:
            stat, pvalue = mannwhitneyu(x, y, axis=0, alternative='two-sided', method='asymptotic', nan_policy='omit')
            mean_a = np.mean(x, axis=0)
            mean_b = np.mean(y, axis=0)
            df = pd.DataFrame(
                {
                    'feature': feature_names,
                    'group_a': a,
                    'group_b': b,
                    'statistic': np.asarray(stat, dtype=float),
                    'pvalue': np.asarray(pvalue, dtype=float),
                    'significant_p': np.asarray(pvalue, dtype=float) < SIGNIFICANCE_THRESHOLD,
                    'mean_a': mean_a,
                    'mean_b': mean_b,
                    'delta_mean': mean_b - mean_a,
                }
            )
            df['fdr'] = benjamini_hochberg(df['pvalue'])
            df = df.sort_values(['pvalue', 'feature'])
        results[f'{a}_vs_{b}'] = df
    return results


def select_pairwise_features_with_hsiclasso(
    feature_df: pd.DataFrame,
    states: pd.Series,
    sig_df: pd.DataFrame,
    group_a: str,
    group_b: str,
    top_n: int = DEFAULT_HSIC_TOP_N,
) -> tuple[list[str], pd.DataFrame]:
    """在 group_a vs group_b 的 p<0.05 基因上，必要时用 HSICLasso 在仅含两组的样本上选 top_n 特征。"""
    candidate_features = sig_df['feature'].astype(str).tolist() if not sig_df.empty else []
    if not candidate_features:
        return [], pd.DataFrame(columns=['feature', 'selection_rank', 'selection_method'])

    if len(candidate_features) <= top_n:
        selected_features = candidate_features
        method = 'all_significant_features'
    else:
        from machine_learning_stage.ML_class import HSICLassoTransformer

        subset_mask = states.isin([group_a, group_b])
        if subset_mask.sum() < 2:
            return [], pd.DataFrame(columns=['feature', 'selection_rank', 'selection_method'])
        X = feature_df.loc[subset_mask, candidate_features].to_numpy(dtype=float)
        y = (states.loc[subset_mask] == group_b).astype(int).to_numpy(dtype=int)
        if len(np.unique(y)) < 2:
            return [], pd.DataFrame(columns=['feature', 'selection_rank', 'selection_method'])
        transformer = HSICLassoTransformer(mode='classification', num_feat=top_n, n_jobs=-1)
        transformer.fit(X, y)
        support = transformer.get_support(indices=True)
        selected_features = [candidate_features[int(idx)] for idx in support]
        method = 'hsiclasso'

    hsic_df = pd.DataFrame(
        {
            'feature': selected_features,
            'selection_rank': np.arange(1, len(selected_features) + 1, dtype=int),
            'selection_method': method,
        }
    )
    return selected_features, hsic_df


def analyze(
    matrix_path: str | Path,
    output_dir: str | Path,
    output_matrix: str | Path | None = None,
    top_n: int = DEFAULT_HSIC_TOP_N,
    pairwise_comparison: str = DEFAULT_PAIRWISE_COMPARISON,
) -> None:
    output_dir = ensure_dir(output_dir)
    differential_dir = ensure_dir(output_dir / 'differential')
    tables_dir = ensure_dir(output_dir / 'tables')
    filtered_matrix_path = Path(output_matrix) if output_matrix else output_dir / 'filtered_matrix.tsv'

    feature_df, states = load_matrix(matrix_path)

    global_diff = run_global_differential(feature_df, states)
    global_diff.loc[global_diff['significant_p']].to_csv(
        differential_dir / 'global_kruskal.tsv', sep='\t', index=False
    )

    pairwise = run_pairwise(feature_df, states)
    pairwise_counts = {}
    for name, df in pairwise.items():
        plot_volcano(df, differential_dir / f'{name}_volcano.png', f'{name} volcano plot')
        pairwise_counts[name] = int(df['significant_p'].sum()) if not df.empty else 0
        if not df.empty:
            sig_df = df.loc[df['significant_p']].copy()
            sig_df.to_csv(differential_dir / f'{name}.tsv', sep='\t', index=False)
        else:
            pd.DataFrame(columns=df.columns if not df.empty else ['feature', 'group_a', 'group_b', 'statistic', 'pvalue', 'significant_p', 'mean_a', 'mean_b', 'delta_mean', 'fdr']).to_csv(
                differential_dir / f'{name}.tsv', sep='\t', index=False
            )

    if pairwise_comparison not in PAIRWISE_SELECTION_CHOICES:
        raise ValueError(
            f'--pairwise-comparison must be one of {PAIRWISE_SELECTION_CHOICES}, got {pairwise_comparison!r}'
        )
    group_a, group_b = parse_pairwise_key(pairwise_comparison)
    pw_df = pairwise.get(pairwise_comparison, pd.DataFrame())
    pw_sig = pw_df.loc[pw_df['pvalue'] < SIGNIFICANCE_THRESHOLD].sort_values(['pvalue', 'feature'])
    n_sig = int(len(pw_sig))
    selected_genes, hsic_df = select_pairwise_features_with_hsiclasso(
        feature_df, states, pw_sig, group_a, group_b, top_n=top_n
    )
    hsic_name = f'{pairwise_comparison}_hsiclasso_selected.tsv'
    hsic_df.to_csv(differential_dir / hsic_name, sep='\t', index=False)
    if 0 < n_sig < top_n:
        print(
            f'warning: only {n_sig} significant genes found for {pairwise_comparison} '
            f'({group_a} vs {group_b}), requested top {top_n}'
        )
    if selected_genes:
        filtered_df = pd.DataFrame({'sample': feature_df.index, 'state': states.reindex(feature_df.index).values})
        filtered_df = pd.concat([filtered_df, feature_df.loc[:, selected_genes].reset_index(drop=True)], axis=1)
        filtered_df.to_csv(filtered_matrix_path, sep='\t', index=False)

    summary = {
        'n_samples': int(feature_df.shape[0]),
        'n_features': int(feature_df.shape[1]),
        'state_counts': states.value_counts().to_dict(),
        'n_global_significant_p_lt_0_05': int(global_diff['significant_p'].sum()),
        'pairwise_significant_counts_p_lt_0_05': pairwise_counts,
        'pairwise_comparison': pairwise_comparison,
        'pairwise_group_a': group_a,
        'pairwise_group_b': group_b,
        'n_pairwise_filtered_genes': int(len(selected_genes)),
        'n_pairwise_p_lt_0_05_total': n_sig,
        'top_n_requested': top_n,
        'hsiclasso_selected_path': str(differential_dir / hsic_name),
        'filtered_matrix_path': str(filtered_matrix_path) if selected_genes else None,
    }
    # 与历史 run_summary 字段兼容（主流程默认 S00_vs_S11 时语义一致）
    if pairwise_comparison == DEFAULT_PAIRWISE_COMPARISON:
        summary['n_s00_s11_filtered_genes'] = summary['n_pairwise_filtered_genes']
        summary['n_s00_s11_p_lt_0_05_total'] = summary['n_pairwise_p_lt_0_05_total']
    with open(tables_dir / 'run_summary.json', 'w', encoding='utf-8') as fh:
        json.dump(summary, fh, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(description='Run differential analysis for one standardized datatype matrix.')
    parser.add_argument('--matrix', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--output-matrix', required=False, default=None)
    parser.add_argument(
        '--top-n',
        type=int,
        default=DEFAULT_HSIC_TOP_N,
        help='在选定 pairwise 上先取 p<0.05，再用 HSICLasso 选 top_n 个特征；若不足则用全部。默认 100。',
    )
    parser.add_argument(
        '--pairwise-comparison',
        default=DEFAULT_PAIRWISE_COMPARISON,
        choices=PAIRWISE_SELECTION_CHOICES,
        help=(
            '用于筛选进入 pseudotime 的差异基因比较组（默认 S00_vs_S11；'
            '可选 S00_vs_S10、S00_vs_S01）'
        ),
    )
    args = parser.parse_args()
    if args.top_n <= 0:
        raise ValueError('--top-n must be a positive int.')
    analyze(
        args.matrix,
        Path(args.output_dir),
        args.output_matrix,
        top_n=args.top_n,
        pairwise_comparison=args.pairwise_comparison,
    )


if __name__ == '__main__':
    main()
