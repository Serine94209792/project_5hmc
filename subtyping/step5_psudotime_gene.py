from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.cluster.hierarchy import linkage
from scipy.stats import spearmanr

try:
    from subtyping.pipeline_utils import ensure_dir, infer_sample_column
except ModuleNotFoundError:
    from pipeline_utils import ensure_dir, infer_sample_column

ALL_STATES = {"S00", "S10", "S01", "S11"}
LM_STATES = {"S00", "S10", "S11"}
VI_STATES = {"S00", "S01", "S11"}
HEATMAP_FIG_WIDTH = 7.2


def benjamini_hochberg(pvals: list[float]) -> list[float]:
    p = np.array(
        [1.0 if (v is None or (isinstance(v, float) and np.isnan(v))) else float(v) for v in pvals],
        dtype=float,
    )
    n = len(p)
    if n == 0:
        return []
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
    out[order] = np.clip(q, 0.0, 1.0)
    return out.tolist()


def load_matrix(path: str | Path) -> tuple[pd.DataFrame, list[str]]:
    df = pd.read_csv(path, sep="\t")
    if "sample" not in df.columns:
        sample_col = infer_sample_column(df)
        df = df.rename(columns={sample_col: "sample"})
    if "state" not in df.columns:
        raise KeyError("Matrix must contain a `state` column.")

    df["sample"] = df["sample"].astype(str)
    df["state"] = df["state"].astype(str)
    dup_mask = df["sample"].duplicated(keep=False)
    if dup_mask.any():
        dup_samples = sorted(df.loc[dup_mask, "sample"].unique().tolist())
        raise ValueError(f"Matrix has duplicated sample IDs: {dup_samples[:10]}")
    feature_cols = [c for c in df.columns if c not in {"sample", "state"}]
    if not feature_cols:
        raise ValueError("No gene columns found in matrix.")
    df[feature_cols] = df[feature_cols].apply(pd.to_numeric, errors="coerce")
    return df, feature_cols


def load_pseudotime(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t")
    required = {"sample", "state", "pseudotime"}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"Pseudotime file missing columns: {sorted(missing)}")
    df["sample"] = df["sample"].astype(str)
    df["state"] = df["state"].astype(str)
    df["pseudotime"] = pd.to_numeric(df["pseudotime"], errors="coerce")
    dup_mask = df["sample"].duplicated(keep=False)
    if dup_mask.any():
        dup_samples = sorted(df.loc[dup_mask, "sample"].unique().tolist())
        raise ValueError(f"Pseudotime file has duplicated sample IDs: {dup_samples[:10]}")
    return df


def load_de_feature_set(path: Path) -> set[str]:
    if not path.exists():
        raise FileNotFoundError(f"Differential file not found: {path}")
    df = pd.read_csv(path, sep="\t")
    gene_col = "feature" if "feature" in df.columns else ("gene" if "gene" in df.columns else None)
    if gene_col is None:
        raise KeyError(f"Cannot find `feature`/`gene` column in {path}")
    return set(df[gene_col].astype(str))


def run_spearman(
    subset_df: pd.DataFrame,
    feature_list: list[str],
) -> pd.DataFrame:
    if len(feature_list) == 0:
        return pd.DataFrame(columns=["gene", "rho", "pvalue", "fdr", "direction"])

    rows = []
    pt = subset_df["pseudotime"].to_numpy(dtype=float)
    for gene in feature_list:
        expr = pd.to_numeric(subset_df[gene], errors="coerce").to_numpy(dtype=float)
        valid = np.isfinite(pt) & np.isfinite(expr)
        if valid.sum() < 3:
            rho, pval = np.nan, np.nan
        else:
            expr_v = expr[valid]
            pt_v = pt[valid]
            if np.nanstd(expr_v) == 0.0 or np.nanstd(pt_v) == 0.0:
                rho, pval = np.nan, np.nan
            else:
                rho, pval = spearmanr(expr_v, pt_v)
        rows.append({"gene": gene, "rho": rho, "pvalue": pval})
    out = pd.DataFrame(rows)
    out["fdr"] = benjamini_hochberg(out["pvalue"].tolist())
    out["direction"] = np.where(
        out["rho"] > 0,
        "up_with_pseudotime",
        np.where(out["rho"] < 0, "down_with_pseudotime", "flat_or_na"),
    )
    return out


def plot_volcano(
    stat_df: pd.DataFrame,
    out_path: Path,
    title: str,
    threshold_value: float,
    threshold_col: str = "fdr",
) -> None:
    eps = 1e-300
    y_threshold = -np.log10(max(threshold_value, eps))
    stat_df = stat_df.copy()
    y_col = f"neg_log10_{threshold_col}"
    threshold_vec = stat_df[threshold_col].fillna(1.0).clip(lower=eps)
    stat_df[y_col] = -np.log10(threshold_vec)
    up_mask = stat_df["significant"] & (stat_df["rho"] > 0)
    down_mask = stat_df["significant"] & (stat_df["rho"] < 0)
    other_mask = ~(up_mask | down_mask)
    sig_n = int(stat_df["significant"].fillna(False).sum())

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(
        stat_df.loc[other_mask, "rho"],
        stat_df.loc[other_mask, y_col],
        s=14,
        alpha=0.55,
        color="#B9B9B9",
        label="not significant",
    )
    ax.scatter(
        stat_df.loc[up_mask, "rho"],
        stat_df.loc[up_mask, y_col],
        s=20,
        alpha=0.85,
        color="#D62728",
        label="up",
    )
    ax.scatter(
        stat_df.loc[down_mask, "rho"],
        stat_df.loc[down_mask, y_col],
        s=20,
        alpha=0.85,
        color="#1F77B4",
        label="down",
    )
    ax.axhline(
        y_threshold,
        color="black",
        linestyle="--",
        linewidth=1.0,
        label=f"-log10({threshold_col}={threshold_value})",
    )
    ax.axvline(0.0, color="gray", linestyle="--", linewidth=1.0)
    ax.set_xlabel("rho")
    ax.set_ylabel(f"-log10({threshold_col})")
    ax.set_title(f"{title} (sig n={sig_n})")
    ax.grid(alpha=0.2)
    ax.legend(frameon=False, loc="best")
    ax.text(
        0.98,
        0.02,
        f"sig n={sig_n}",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=10,
        bbox={"boxstyle": "round,pad=0.2", "facecolor": "white", "alpha": 0.75, "edgecolor": "none"},
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def build_pseudotime_col_colors(pseudotime: pd.Series) -> list[tuple[float, float, float, float]]:
    pt = pd.to_numeric(pseudotime, errors="coerce").to_numpy(dtype=float)
    if len(pt) == 0:
        return []
    finite = np.isfinite(pt)
    if not finite.any():
        return [mpl.colormaps["viridis"](0.0)] * len(pt)
    vmin = float(np.nanmin(pt[finite]))
    vmax = float(np.nanmax(pt[finite]))
    if np.isclose(vmin, vmax):
        normed = np.full(len(pt), 0.5, dtype=float)
    else:
        normed = np.clip((pt - vmin) / (vmax - vmin), 0.0, 1.0)
    normed[~finite] = 0.0
    cmap = mpl.colormaps["viridis"]
    return [cmap(v) for v in normed]


def plot_heatmap(
    subset_df: pd.DataFrame,
    genes: list[str],
    out_path: Path,
    title: str,
) -> None:
    if len(genes) == 0:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(0.5, 0.5, "No significant genes", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        fig.tight_layout()
        fig.savefig(out_path, dpi=220)
        plt.close(fig)
        return

    ordered = subset_df.sort_values("pseudotime", ascending=True).copy()
    expr = ordered[genes].apply(pd.to_numeric, errors="coerce").fillna(0.0)

    mean = expr.mean(axis=0)
    std = expr.std(axis=0, ddof=0).replace(0.0, 1.0)
    scaled = (expr - mean) / std

    plot_data = scaled.T
    pseudotime_colors = build_pseudotime_col_colors(ordered["pseudotime"])
    row_linkage = linkage(plot_data.to_numpy(dtype=float), method="average", metric="euclidean")

    fig_w = HEATMAP_FIG_WIDTH
    fig_h = max(5, min(20, plot_data.shape[0] * 0.18))
    g = sns.clustermap(
        plot_data,
        cmap="RdBu_r",
        center=0.0,
        row_cluster=True,
        col_cluster=False,
        row_linkage=row_linkage,
        col_colors=pseudotime_colors,
        xticklabels=False,
        yticklabels=False,
        figsize=(fig_w, fig_h),
        dendrogram_ratio=(0.001, 0.001),
        cbar_kws={"label": "z-score"},
    )
    g.ax_row_dendrogram.set_visible(False)
    if hasattr(g, "ax_col_dendrogram"):
        g.ax_col_dendrogram.set_visible(False)
    if hasattr(g, "ax_col_colors"):
        g.ax_col_colors.set_ylabel("")
        g.ax_col_colors.set_yticks([])
        g.ax_col_colors.set_xticks([])
        g.ax_col_colors.text(
            0.5,
            1.15,
            "Pseudotime",
            transform=g.ax_col_colors.transAxes,
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
        )
    g.ax_heatmap.set_xlabel("Pseudotime")
    g.ax_heatmap.set_ylabel("")
    g.ax_heatmap.set_title("")
    g.ax_heatmap.set_xticks([])
    g.ax_heatmap.tick_params(axis="x", bottom=False, labelbottom=False)
    g.ax_row_dendrogram.set_xticks([])
    g.ax_row_dendrogram.set_yticks([])
    g.fig.savefig(out_path, dpi=220)
    plt.close(g.fig)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Step5: pseudotime signatures for LM/VI paths and global set.")
    p.add_argument("--matrix", required=True, help="Filtered matrix TSV (for diff_features_all).")
    p.add_argument("--full-matrix", required=True, help="Full matrix TSV (for LM/VI path candidate expression).")
    p.add_argument("--pseudotime", required=True, help="Pseudotime TSV (must include sample/state/pseudotime).")
    p.add_argument("--output-dir", required=True, help="Datatype output root directory.")
    p.add_argument("--fdr-threshold", type=float, default=0.05)
    p.add_argument("--min-abs-rho", type=float, default=0.0)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = ensure_dir(Path(args.output_dir) / "trajectory_genes")
    legacy_files = [
        out_dir / "summary.json",
        out_dir / "diff_features_all_sets.tsv",
        out_dir / "diff_features_lm_path.tsv",
        out_dir / "diff_features_vi_path.tsv",
        out_dir / "volcano_all_sets.png",
        out_dir / "volcano_lm_path.png",
        out_dir / "volcano_vi_path.png",
    ]
    for legacy in legacy_files:
        if legacy.exists():
            legacy.unlink()

    matrix_df, feature_cols = load_matrix(args.matrix)
    full_matrix_df, full_feature_cols = load_matrix(args.full_matrix)
    full_feature_set = set(full_feature_cols)
    pt_df = load_pseudotime(args.pseudotime)

    def merge_with_pt(mat: pd.DataFrame) -> pd.DataFrame:
        m = mat.merge(
            pt_df[["sample", "state", "pseudotime"]],
            on="sample",
            how="inner",
            suffixes=("_matrix", "_pt"),
        )
        if m.empty:
            raise ValueError("No overlapping samples between matrix and pseudotime files.")
        if "state_matrix" in m.columns and "state_pt" in m.columns:
            mismatch = m["state_matrix"] != m["state_pt"]
            if mismatch.any():
                raise ValueError("State mismatch between matrix and pseudotime for some samples.")
            m["state"] = m["state_matrix"]
        elif "state" not in m.columns:
            raise KeyError("Merged table missing state information.")
        return m

    merged = merge_with_pt(matrix_df)
    merged_full = merge_with_pt(full_matrix_df)

    diff_dir = Path(args.output_dir) / "differential"
    lm_candidates_raw = load_de_feature_set(diff_dir / "S00_vs_S10.tsv") & load_de_feature_set(
        diff_dir / "S10_vs_S11.tsv"
    )
    vi_candidates_raw = load_de_feature_set(diff_dir / "S00_vs_S01.tsv") & load_de_feature_set(
        diff_dir / "S01_vs_S11.tsv"
    )
    lm_candidates = sorted([g for g in lm_candidates_raw if g in full_feature_set])
    vi_candidates = sorted([g for g in vi_candidates_raw if g in full_feature_set])
    print(f"[step5] LM_path 初始交集特征数: {len(lm_candidates_raw)} → 在 full-matrix 中匹配: {len(lm_candidates)}")
    print(f"[step5] VI_path 初始交集特征数: {len(vi_candidates_raw)} → 在 full-matrix 中匹配: {len(vi_candidates)}")

    all_df = run_spearman(
        subset_df=merged.loc[merged["state"].isin(ALL_STATES)].copy(),
        feature_list=feature_cols,
    )
    all_df["significant"] = (all_df["fdr"] < args.fdr_threshold) & (all_df["rho"].abs() >= args.min_abs_rho)
    diff_features_all = all_df.loc[all_df["significant"], ["gene", "rho", "pvalue", "fdr", "direction"]].sort_values(
        ["fdr", "rho"], ascending=[True, False]
    )
    diff_features_all.to_csv(out_dir / "diff_features_all.tsv", sep="\t", index=False)
    plot_volcano(
        stat_df=all_df,
        out_path=out_dir / "volcano_diff_features_all.png",
        title="Volcano (all samples)",
        threshold_value=args.fdr_threshold,
        threshold_col="fdr",
    )
    all_sig_genes = diff_features_all["gene"].astype(str).tolist()
    plot_heatmap(
        subset_df=merged.loc[merged["state"].isin(ALL_STATES)].copy(),
        genes=all_sig_genes,
        out_path=out_dir / "heatmap_diff_features_all_sig.png",
        title="All-sample significant features across pseudotime",
    )

    lm_df = run_spearman(
        subset_df=merged_full.loc[merged_full["state"].isin(LM_STATES)].copy(),
        feature_list=lm_candidates,
    )
    lm_df["significant"] = lm_df["fdr"] < args.fdr_threshold
    lm_sig = lm_df.loc[lm_df["significant"], ["gene", "rho", "pvalue", "fdr", "direction"]].sort_values(
        ["fdr", "rho"], ascending=[True, False]
    )
    lm_sig.to_csv(out_dir / "LM_path_signature.tsv", sep="\t", index=False)
    plot_volcano(
        stat_df=lm_df,
        out_path=out_dir / "volcano_LM_path_signature.png",
        title="Volcano (LM path signature candidates)",
        threshold_value=args.fdr_threshold,
        threshold_col="fdr",
    )
    plot_heatmap(
        subset_df=merged_full.loc[merged_full["state"].isin(LM_STATES)].copy(),
        genes=lm_sig["gene"].astype(str).tolist(),
        out_path=out_dir / "heatmap_LM_path_signature.png",
        title="LM path signature across pseudotime",
    )

    vi_df = run_spearman(
        subset_df=merged_full.loc[merged_full["state"].isin(VI_STATES)].copy(),
        feature_list=vi_candidates,
    )
    vi_df["significant"] = vi_df["fdr"] < args.fdr_threshold
    vi_sig = vi_df.loc[vi_df["significant"], ["gene", "rho", "pvalue", "fdr", "direction"]].sort_values(
        ["fdr", "rho"], ascending=[True, False]
    )
    vi_sig.to_csv(out_dir / "VI_path_signature.tsv", sep="\t", index=False)
    plot_volcano(
        stat_df=vi_df,
        out_path=out_dir / "volcano_VI_path_signature.png",
        title="Volcano (VI path signature candidates)",
        threshold_value=args.fdr_threshold,
        threshold_col="fdr",
    )
    plot_heatmap(
        subset_df=merged_full.loc[merged_full["state"].isin(VI_STATES)].copy(),
        genes=vi_sig["gene"].astype(str).tolist(),
        out_path=out_dir / "heatmap_VI_path_signature.png",
        title="VI path signature across pseudotime",
    )


if __name__ == "__main__":
    main()
