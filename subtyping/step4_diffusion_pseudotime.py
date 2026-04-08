from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path

import anndata as ad
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns
from matplotlib.patches import Ellipse
from scipy.spatial.distance import cdist
from scipy.stats import chi2, kruskal, mannwhitneyu

from subtyping.pipeline_utils import STATE_ORDER, ensure_dir, infer_sample_column

sns.set_style("white")

STATE_PALETTE = dict(zip(STATE_ORDER, sns.color_palette("tab10", n_colors=len(STATE_ORDER))))

DIFFUSION_FIGSIZE = (7.6, 7.6)
PSEUDOTIME_FIGSIZE = (11.5, 5.8)
AXIS_LABEL_FONTSIZE = 17
TICK_LABEL_FONTSIZE = 13
LEGEND_FONTSIZE = 14
TITLE_FONTSIZE = 16
ANNOTATION_FONTSIZE = 13
CENTROID_MARKER_SIZE = 175
POINT_ALPHA = 0.82


def benjamini_hochberg(pvals: list[float]) -> list[float]:
    p = np.array([1.0 if (v is None or (isinstance(v, float) and np.isnan(v))) else float(v) for v in pvals])
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


def load_matrix(path: str | Path) -> tuple[np.ndarray, np.ndarray, pd.Series, list[str]]:
    df = pd.read_csv(path, sep="\t")
    if "sample" not in df.columns:
        sample_col = infer_sample_column(df)
        df = df.rename(columns={sample_col: "sample"})
    if "state" not in df.columns:
        raise KeyError("Matrix must contain a `state` column.")

    df["sample"] = df["sample"].astype(str)
    df["state"] = df["state"].astype(str)

    feature_cols = [c for c in df.columns if c not in {"sample", "state"}]
    if len(feature_cols) == 0:
        raise ValueError("No feature columns found in matrix.")

    X = df[feature_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(dtype=float)
    samples = df["sample"].to_numpy()
    states = df.set_index("sample")["state"]
    return X, samples, states, feature_cols


def compute_root_medoid(diff_coords: np.ndarray, states: pd.Series, root_state: str = "S00") -> tuple[int, str]:
    mask = states.values == root_state
    if mask.sum() == 0:
        raise ValueError(f"No samples found for root_state={root_state!r}.")
    root_coords = diff_coords[mask]
    # Medoid: minimize sum of distances within root_state group.
    D = cdist(root_coords, root_coords, metric="euclidean")
    medoid_local = int(np.argmin(D.sum(axis=1)))
    root_global_idx = int(np.where(mask)[0][medoid_local])
    return root_global_idx, str(states.index[root_global_idx])


def compute_state_centroids(coords_df: pd.DataFrame) -> pd.DataFrame:
    """Return one row per state with mean dc1, dc2. coords_df must have dc1, dc2, state."""
    out = []
    for s in STATE_ORDER:
        sub = coords_df.loc[coords_df["state"] == s]
        if len(sub) == 0:
            continue
        out.append({"state": s, "dc1": sub["dc1"].mean(), "dc2": sub["dc2"].mean()})
    return pd.DataFrame(out)


def _safe_covariance(coords: np.ndarray) -> np.ndarray:
    if coords.shape[0] <= 1:
        return np.eye(2) * 1e-6
    cov = np.cov(coords, rowvar=False)
    cov = np.asarray(cov, dtype=float)
    if cov.shape != (2, 2):
        return np.eye(2) * 1e-6
    cov += np.eye(2) * 1e-6
    return cov


def build_state_annotations(coords_df: pd.DataFrame, coverage: float = 0.9) -> pd.DataFrame:
    """Compute robust state centers and ellipse parameters for dc1/dc2 plotting."""
    out = []
    coverage_scale = float(np.sqrt(chi2.ppf(coverage, df=2)))

    for state in STATE_ORDER:
        sub = coords_df.loc[coords_df["state"] == state, ["dc1", "dc2"]]
        if sub.empty:
            continue

        coords = sub.to_numpy(dtype=float)
        center = np.median(coords, axis=0)

        if coords.shape[0] >= 4:
            dists = np.linalg.norm(coords - center, axis=1)
            keep_threshold = float(np.quantile(dists, 0.85))
            inlier_coords = coords[dists <= keep_threshold]
            if inlier_coords.shape[0] < 3:
                inlier_coords = coords
        else:
            inlier_coords = coords

        cov = _safe_covariance(inlier_coords)
        eigvals, eigvecs = np.linalg.eigh(cov)
        order = np.argsort(eigvals)[::-1]
        eigvals = np.clip(eigvals[order], 1e-6, None)
        eigvecs = eigvecs[:, order]

        angle = float(np.degrees(np.arctan2(eigvecs[1, 0], eigvecs[0, 0])))
        ellipse_width = float(2 * coverage_scale * np.sqrt(eigvals[0]))
        ellipse_height = float(2 * coverage_scale * np.sqrt(eigvals[1]))

        out.append(
            {
                "state": state,
                "center_dc1": float(center[0]),
                "center_dc2": float(center[1]),
                "ellipse_width": ellipse_width,
                "ellipse_height": ellipse_height,
                "ellipse_angle": angle,
            }
        )
    annotations = pd.DataFrame(out)
    if annotations.empty:
        return annotations
    return _assign_label_positions(annotations, coords_df)


def _assign_label_positions(annotations: pd.DataFrame, coords_df: pd.DataFrame) -> pd.DataFrame:
    annotations = annotations.copy()
    x_min = float(coords_df["dc1"].min())
    x_max = float(coords_df["dc1"].max())
    y_min = float(coords_df["dc2"].min())
    y_max = float(coords_df["dc2"].max())
    x_span = max(float(coords_df["dc1"].max() - coords_df["dc1"].min()), 1e-6)
    y_span = max(float(coords_df["dc2"].max() - coords_df["dc2"].min()), 1e-6)
    min_sep = 0.03 * np.hypot(x_span, y_span)
    x_margin = 0.04 * x_span
    y_margin = 0.04 * y_span

    chosen: list[np.ndarray] = []
    for idx, row in annotations.iterrows():
        center = np.array([row["center_dc1"], row["center_dc2"]], dtype=float)
        dx = max(row["ellipse_width"] * 0.22, x_span * 0.015)
        dy = max(row["ellipse_height"] * 0.22, y_span * 0.015)
        candidates = [
            center + np.array([0.65 * dx, 0.65 * dy]),
            center + np.array([0.65 * dx, -0.65 * dy]),
            center + np.array([-0.65 * dx, 0.65 * dy]),
            center + np.array([-0.65 * dx, -0.65 * dy]),
            center + np.array([dx, dy]),
            center + np.array([dx, -dy]),
            center + np.array([-dx, dy]),
            center + np.array([-dx, -dy]),
            center + np.array([0.0, 1.05 * dy]),
            center + np.array([0.0, -1.05 * dy]),
            center + np.array([1.05 * dx, 0.0]),
            center + np.array([-1.05 * dx, 0.0]),
        ]

        best = candidates[0]
        best_score = -np.inf
        for cand in candidates:
            cand = np.array(
                [
                    np.clip(cand[0], x_min + x_margin, x_max - x_margin),
                    np.clip(cand[1], y_min + y_margin, y_max - y_margin),
                ]
            )
            if chosen:
                dists = [float(np.linalg.norm(cand - prev)) for prev in chosen]
                min_dist = min(dists)
            else:
                min_dist = np.inf
            center_dist = float(np.linalg.norm(cand - center))
            penalty = 0.09 * center_dist
            bonus = 2.0 if min_dist >= min_sep else min_dist / max(min_sep, 1e-6)
            score = bonus - penalty
            if score > best_score:
                best = cand
                best_score = score

        annotations.loc[idx, "label_dc1"] = float(best[0])
        annotations.loc[idx, "label_dc2"] = float(best[1])
        chosen.append(best)
    return annotations


def add_state_annotations(ax: plt.Axes, coords_df: pd.DataFrame, show_ellipses: bool = True) -> None:
    annotations = build_state_annotations(coords_df)
    if annotations.empty:
        return

    if show_ellipses:
        for _, row in annotations.iterrows():
            color = STATE_PALETTE.get(row["state"], "#999999")
            ellipse = Ellipse(
                (row["center_dc1"], row["center_dc2"]),
                width=row["ellipse_width"],
                height=row["ellipse_height"],
                angle=row["ellipse_angle"],
                facecolor=color,
                edgecolor=color,
                alpha=0.18,
                linewidth=1.2,
                zorder=1.5,
            )
            ax.add_patch(ellipse)

    ax.scatter(
        annotations["center_dc1"],
        annotations["center_dc2"],
        marker="o",
        s=CENTROID_MARKER_SIZE,
        c=[STATE_PALETTE.get(state, "#999999") for state in annotations["state"]],
        edgecolors="white",
        linewidths=1.8,
        zorder=5,
    )

    for _, row in annotations.iterrows():
        ax.plot(
            [row["center_dc1"], row["label_dc1"]],
            [row["center_dc2"], row["label_dc2"]],
            color="#666666",
            linewidth=0.7,
            alpha=0.7,
            zorder=5.5,
        )
        txt = ax.text(
            row["label_dc1"],
            row["label_dc2"],
            row["state"],
            fontsize=ANNOTATION_FONTSIZE,
            fontweight="bold",
            color="#222222",
            ha="left",
            va="center",
            zorder=6,
            bbox={"boxstyle": "round,pad=0.22", "facecolor": "white", "edgecolor": "none", "alpha": 0.82},
        )
        txt.set_path_effects([pe.withStroke(linewidth=2.2, foreground="white")])


def ensure_dc_columns(df: pd.DataFrame, n_dcs: int) -> list[str]:
    cols = [f"dc{i}" for i in range(1, n_dcs + 1)]
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"Missing diffusion columns: {missing}")
    return cols


def style_diffusion_axis(ax: plt.Axes, xlabel: str = "DC1", ylabel: str = "DC2") -> None:
    ax.grid(False)
    ax.set_box_aspect(1)
    ax.set_xlabel(xlabel, fontsize=AXIS_LABEL_FONTSIZE, fontweight="bold")
    ax.set_ylabel(ylabel, fontsize=AXIS_LABEL_FONTSIZE, fontweight="bold")
    ax.set_title("")
    ax.tick_params(
        axis="both",
        which="both",
        bottom=False,
        left=False,
        labelbottom=False,
        labelleft=False,
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_visible(False)


def _style_legend(legend: plt.Legend | None) -> None:
    if legend is None:
        return
    legend.set_title(None)
    frame = legend.get_frame()
    frame.set_facecolor("white")
    frame.set_alpha(0.12)
    frame.set_edgecolor("#d0d0d0")
    for text in legend.get_texts():
        text.set_fontsize(LEGEND_FONTSIZE)
        text.set_fontweight("bold")
    if hasattr(legend, "legend_handles"):
        for handle in legend.legend_handles:
            if hasattr(handle, "set_sizes"):
                handle.set_sizes([90])


def plot_diffusion(coords_df: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=DIFFUSION_FIGSIZE)
    add_state_annotations(ax, coords_df, show_ellipses=True)
    sns.scatterplot(
        data=coords_df,
        x="dc1",
        y="dc2",
        hue="state",
        hue_order=STATE_ORDER,
        palette=STATE_PALETTE,
        ax=ax,
        s=46,
        alpha=POINT_ALPHA,
        linewidth=0,
        zorder=3,
    )
    style_diffusion_axis(ax, xlabel="DC1", ylabel="DC2")
    _style_legend(ax.legend_)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_diffusion_pseudotime_gradient(coords_df: pd.DataFrame, out_path: Path) -> None:
    """coords_df must have dc1, dc2, pseudotime, and state (for centroid annotation)."""
    fig, ax = plt.subplots(figsize=DIFFUSION_FIGSIZE)
    if "state" in coords_df.columns:
        add_state_annotations(ax, coords_df, show_ellipses=False)
    sca = ax.scatter(
        coords_df["dc1"],
        coords_df["dc2"],
        c=coords_df["pseudotime"],
        cmap="viridis",
        s=42,
        alpha=POINT_ALPHA,
        lw=0,
    )
    cbar = fig.colorbar(sca, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Pseudotime", fontsize=AXIS_LABEL_FONTSIZE, fontweight="bold")
    cbar.ax.tick_params(labelsize=TICK_LABEL_FONTSIZE)
    if "state" in coords_df.columns:
        pt_df = coords_df[["state", "pseudotime"]].copy()
        pairwise_stats = compute_pairwise_pseudotime_stats(pt_df)
        global_stat = compute_global_pseudotime_stat(pt_df)
        annotate_diffusion_pseudotime_stats(ax, global_stat, pairwise_stats)
    style_diffusion_axis(ax, xlabel="DC1", ylabel="DC2")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_pseudotime_violin(pt_df: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.2, 5.8))

    sns.violinplot(
        data=pt_df,
        x="state",
        y="pseudotime",
        hue="state",
        order=STATE_ORDER,
        palette=STATE_PALETTE,
        legend=False,
        inner=None,
        cut=0,
        linewidth=1.0,
        saturation=1.0,
        width=0.72,
        ax=ax,
    )
    sns.boxplot(
        data=pt_df,
        x="state",
        y="pseudotime",
        hue="state",
        order=STATE_ORDER,
        palette=STATE_PALETTE,
        legend=False,
        width=0.26,
        showcaps=True,
        showfliers=False,
        boxprops={"alpha": 0.95, "linewidth": 1.1},
        whiskerprops={"linewidth": 1.0},
        medianprops={"color": "#1f1f1f", "linewidth": 1.4},
        ax=ax,
    )
    sns.stripplot(
        data=pt_df,
        x="state",
        y="pseudotime",
        order=STATE_ORDER,
        hue="state",
        palette=STATE_PALETTE,
        dodge=False,
        alpha=0.45,
        size=3.2,
        jitter=0.15,
        linewidth=0,
        ax=ax,
    )
    if ax.legend_ is not None:
        ax.legend_.remove()

    pairwise_stats = compute_pairwise_pseudotime_stats(pt_df)
    global_stat = compute_global_pseudotime_stat(pt_df)
    annotate_global_pseudotime_stat(ax, global_stat)

    add_pairwise_significance_annotations(ax, pt_df, pairwise_stats)
    ax.set_title("")
    ax.set_xlabel("")
    ax.set_ylabel("Pseudotime", fontsize=AXIS_LABEL_FONTSIZE, fontweight="bold")
    ax.tick_params(axis="x", labelsize=TICK_LABEL_FONTSIZE)
    ax.tick_params(axis="y", labelsize=TICK_LABEL_FONTSIZE)
    for label in ax.get_xticklabels():
        label.set_fontweight("bold")
    ax.grid(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def compute_global_pseudotime_stat(pt_df: pd.DataFrame) -> dict[str, float] | None:
    state_to_pt = {
        s: pd.to_numeric(pt_df.loc[pt_df["state"] == s, "pseudotime"], errors="coerce").dropna().to_numpy(dtype=float)
        for s in STATE_ORDER
    }
    groups = [state_to_pt[s] for s in STATE_ORDER if len(state_to_pt[s]) >= 1]
    if len(groups) < 2 or sum(len(g) for g in groups) <= 2:
        return None
    stat_kw, p_kw = kruskal(*groups)
    return {"statistic": float(stat_kw), "pvalue": float(p_kw)}


def compute_pairwise_pseudotime_stats(pt_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    pvals = []
    valid_states = [s for s in STATE_ORDER if s in set(pt_df["state"].astype(str))]
    for group1, group2 in combinations(valid_states, 2):
        g1 = pd.to_numeric(pt_df.loc[pt_df["state"] == group1, "pseudotime"], errors="coerce").dropna().to_numpy(dtype=float)
        g2 = pd.to_numeric(pt_df.loc[pt_df["state"] == group2, "pseudotime"], errors="coerce").dropna().to_numpy(dtype=float)
        if len(g1) == 0 or len(g2) == 0:
            stat, pvalue = np.nan, np.nan
        else:
            stat, pvalue = mannwhitneyu(g1, g2, alternative="two-sided")
        rows.append({"group1": group1, "group2": group2, "statistic": stat, "pvalue": pvalue})
        pvals.append(pvalue)

    out = pd.DataFrame(rows)
    out["fdr"] = benjamini_hochberg(pvals)
    out["significant"] = out["fdr"] < 0.05
    return out


def compute_pairwise_pseudotime_matrix(pt_df: pd.DataFrame) -> pd.DataFrame:
    pairwise_stats = compute_pairwise_pseudotime_stats(pt_df)
    present_states = [s for s in STATE_ORDER if s in set(pt_df["state"].astype(str))]
    matrix = pd.DataFrame(np.nan, index=present_states, columns=present_states, dtype=float)
    for _, row in pairwise_stats.iterrows():
        g1 = str(row["group1"])
        g2 = str(row["group2"])
        matrix.loc[g1, g2] = float(row["fdr"])
        matrix.loc[g2, g1] = float(row["fdr"])
    return matrix


def summarize_significant_pairwise_stats(pairwise_stats: pd.DataFrame) -> str:
    sig_df = pairwise_stats.loc[pairwise_stats["significant"]].copy()
    if sig_df.empty:
        return ""
    sig_df = sig_df.sort_values(["fdr", "group1", "group2"]).reset_index(drop=True)
    lines = ["Sig. pair:"]
    for _, row in sig_df.iterrows():
        lines.append(f"{row['group1']} vs {row['group2']} {_pvalue_to_stars(float(row['fdr']))}")
    return "\n".join(lines)


def _pvalue_to_stars(pvalue: float) -> str:
    if not np.isfinite(pvalue):
        return "ns"
    if pvalue < 0.001:
        return "***"
    if pvalue < 0.01:
        return "**"
    if pvalue < 0.05:
        return "*"
    return "ns"


def add_pairwise_significance_annotations(ax: plt.Axes, pt_df: pd.DataFrame, pairwise_stats: pd.DataFrame) -> None:
    sig_df = pairwise_stats.loc[pairwise_stats["significant"]].copy()
    if sig_df.empty:
        return

    x_pos = {state: idx for idx, state in enumerate(STATE_ORDER)}
    y_max = float(pd.to_numeric(pt_df["pseudotime"], errors="coerce").max())
    y_min = float(pd.to_numeric(pt_df["pseudotime"], errors="coerce").min())
    y_span = max(y_max - y_min, 1e-6)
    level_step = 0.08 * y_span
    base_y = y_max + 0.06 * y_span

    sig_df = sig_df.assign(width=sig_df.apply(lambda row: x_pos[row["group2"]] - x_pos[row["group1"]], axis=1))
    sig_df = sig_df.sort_values(["width", "fdr"], ascending=[True, True]).reset_index(drop=True)

    used_levels: list[tuple[int, int, int]] = []
    for _, row in sig_df.iterrows():
        start = x_pos[row["group1"]]
        end = x_pos[row["group2"]]
        level = 0
        while any(not (end < s or start > e) and lvl == level for s, e, lvl in used_levels):
            level += 1
        used_levels.append((start, end, level))
        y = base_y + level * level_step

        ax.plot([start, start, end, end], [y - 0.015 * y_span, y, y, y - 0.015 * y_span], color="black", linewidth=1.0)
        ax.text(
            (start + end) / 2,
            y + 0.01 * y_span,
            _pvalue_to_stars(float(row["fdr"])),
            ha="center",
            va="bottom",
            fontsize=ANNOTATION_FONTSIZE,
            fontweight="bold",
        )

    ax.set_ylim(y_min - 0.02 * y_span, base_y + (max(level for _, _, level in used_levels) + 1.6) * level_step)


def annotate_global_pseudotime_stat(ax: plt.Axes, global_stat: dict[str, float] | None) -> None:
    if global_stat is None:
        return
    text = f"Kruskal-Wallis: p = {global_stat['pvalue']:.2e}"
    ax.text(
        0.02,
        0.98,
        text,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=ANNOTATION_FONTSIZE,
        fontweight="bold",
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": "#bdbdbd", "alpha": 0.92},
    )


def annotate_diffusion_pseudotime_stats(
    ax: plt.Axes,
    global_stat: dict[str, float] | None,
    pairwise_stats: pd.DataFrame,
) -> None:
    summary = summarize_significant_pairwise_stats(pairwise_stats)
    if not summary:
        return

    ax.text(
        0.02,
        0.98,
        summary,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=ANNOTATION_FONTSIZE - 1,
        fontweight="bold",
        color="#222222",
        bbox={"boxstyle": "round,pad=0.28", "facecolor": "white", "edgecolor": "#c7c7c7", "alpha": 0.92},
        zorder=8,
    )


def plot_pairwise_pseudotime_heatmap(ax: plt.Axes, pt_df: pd.DataFrame, pairwise_stats: pd.DataFrame) -> None:
    matrix = compute_pairwise_pseudotime_matrix(pt_df)
    if matrix.empty:
        ax.axis("off")
        return

    annot = matrix.copy().astype(object)
    for i in matrix.index:
        for j in matrix.columns:
            value = matrix.loc[i, j]
            if i == j or not np.isfinite(value):
                annot.loc[i, j] = ""
            else:
                annot.loc[i, j] = _pvalue_to_stars(float(value))

    mask = np.triu(np.ones_like(matrix, dtype=bool))
    sns.heatmap(
        matrix,
        mask=mask,
        cmap="Reds_r",
        vmin=0.0,
        vmax=max(float(np.nanmax(matrix.to_numpy())), 0.05),
        square=True,
        linewidths=1.0,
        linecolor="white",
        cbar_kws={"label": "Pairwise FDR"},
        annot=annot,
        fmt="",
        annot_kws={"fontsize": ANNOTATION_FONTSIZE, "fontweight": "bold"},
        ax=ax,
    )
    ax.set_title("Pairwise\nMann-Whitney", fontsize=TITLE_FONTSIZE - 1, fontweight="bold")
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.tick_params(axis="x", labelsize=TICK_LABEL_FONTSIZE, rotation=0)
    ax.tick_params(axis="y", labelsize=TICK_LABEL_FONTSIZE, rotation=0)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontweight("bold")
    if ax.collections and hasattr(ax.collections[0], "colorbar") and ax.collections[0].colorbar is not None:
        cbar = ax.collections[0].colorbar
        cbar.ax.tick_params(labelsize=TICK_LABEL_FONTSIZE)
        cbar.set_label("Pairwise FDR", fontsize=AXIS_LABEL_FONTSIZE - 1, fontweight="bold")


def run_step(
    matrix_path: str | Path,
    output_dir: str | Path,
    k: int,
    n_dcs: int,
    random_state: int,
    root_state: str = "S00",
    scale_before_pca: bool = True,
) -> None:
    output_dir = ensure_dir(output_dir)
    traj_dir = ensure_dir(output_dir / "trajectory")

    X, samples, states_series, feature_cols = load_matrix(matrix_path)

    if X.shape[0] < 3:
        raise ValueError("Need at least 3 samples for diffusion pseudotime.")

    # scanpy diffmap requires n_comps > 2
    n_dcs = int(max(3, min(n_dcs, X.shape[0] - 1)))

    obs = pd.DataFrame({"state": states_series.values}, index=samples)
    adata = ad.AnnData(X=X, obs=obs)
    adata.obs_names = samples
    adata.var_names = feature_cols

    if scale_before_pca:
        sc.pp.scale(adata, zero_center=True, max_value=10)
    sc.pp.pca(adata, n_comps=min(50, X.shape[1] - 1), random_state=random_state)
    sc.pp.neighbors(adata, n_neighbors=k, use_rep="X_pca", random_state=random_state)
    sc.tl.diffmap(adata, n_comps=n_dcs, random_state=random_state)

    diff_coords = adata.obsm["X_diffmap"]
    if diff_coords.shape[1] < 2:
        raise RuntimeError("Diffusion coordinates must have at least 2 dimensions for plotting.")

    coords_df = pd.DataFrame(diff_coords[:, :n_dcs], columns=[f"dc{i}" for i in range(1, n_dcs + 1)])
    coords_df.insert(0, "sample", samples)
    coords_df.insert(1, "state", states_series.values)

    diffusion_coordinates_path = traj_dir / "diffusion_coordinates.tsv"
    coords_df.to_csv(diffusion_coordinates_path, sep="\t", index=False)

    root_idx, root_sample = compute_root_medoid(diff_coords[:, :n_dcs], states_series, root_state=root_state)
    adata.uns["iroot"] = int(root_idx)

    sc.tl.dpt(adata, n_dcs=n_dcs)
    pseudotime = adata.obs["dpt_pseudotime"].to_numpy(dtype=float)
    if not np.isfinite(pseudotime).all():
        raise RuntimeError("scanpy.tl.dpt produced non-finite pseudotime values.")

    pt_df = pd.DataFrame(
        {
            "sample": samples,
            "state": states_series.values,
            "pseudotime": pseudotime,
            "root_sample": root_sample,
            "k": int(k),
        }
    )
    pt_path = traj_dir / "pseudotime.tsv"
    pt_df.to_csv(pt_path, sep="\t", index=False)

    # Plots
    plot_diffusion(coords_df, traj_dir / "diffusion_plot.png")

    coords_df_pt = coords_df.merge(pt_df[["sample", "pseudotime"]], on="sample", how="left")
    plot_diffusion_pseudotime_gradient(coords_df_pt, traj_dir / "diffusion_pseudotime_gradient.png")
    plot_pseudotime_violin(pt_df, traj_dir / "pseudotime_violin.png")

    state_to_pt = {s: pseudotime[states_series.values == s] for s in STATE_ORDER}
    groups = [state_to_pt[s] for s in STATE_ORDER if len(state_to_pt[s]) >= 1]

    global_kw = None
    if len(groups) >= 2 and sum(len(g) for g in groups) > 2:
        stat_kw, p_kw = kruskal(*[g for g in groups])
        global_kw = {"statistic": float(stat_kw), "pvalue": float(p_kw)}

    directional_tests: list[dict] = []

    def safe_mwu(g1: np.ndarray, g2: np.ndarray, alternative: str) -> tuple[float, float]:
        if len(g1) == 0 or len(g2) == 0:
            return np.nan, np.nan
        if len(g1) < 2 or len(g2) < 2:
            # mannwhitneyu still works, but we keep it.
            pass
        stat, p = mannwhitneyu(g1, g2, alternative=alternative)
        return float(stat), float(p)

    s00 = state_to_pt["S00"]
    # Hypothesis: S00 < others
    for target_state, alternative in [("S10", "less"), ("S01", "less"), ("S11", "less")]:
        stat, p = safe_mwu(s00, state_to_pt[target_state], alternative=alternative)
        directional_tests.append(
            {
                "comparison": f"S00_vs_{target_state}",
                "alternative": alternative,
                "statistic": stat,
                "pvalue": p,
            }
        )
    # Hypothesis: S11 > single-positive states
    for target_state, alternative in [("S10", "greater"), ("S01", "greater")]:
        stat, p = safe_mwu(state_to_pt["S11"], state_to_pt[target_state], alternative=alternative)
        directional_tests.append(
            {
                "comparison": f"S11_vs_{target_state}",
                "alternative": alternative,
                "statistic": stat,
                "pvalue": p,
            }
        )

    pvals_dir = [t["pvalue"] for t in directional_tests if np.isfinite(t["pvalue"])]
    fdr_dir_values = benjamini_hochberg(pvals_dir) if len(pvals_dir) else []
    dir_fdr_map = {}
    j = 0
    for t in directional_tests:
        if np.isfinite(t["pvalue"]):
            dir_fdr_map[id(t)] = fdr_dir_values[j]
            j += 1
    for t in directional_tests:
        if "fdr" not in t:
            t["fdr"] = float(dir_fdr_map.get(id(t), np.nan))

    stats_obj = {
        "k": int(k),
        "n_dcs": int(n_dcs),
        "root_state": root_state,
        "root_sample": root_sample,
        "scale_before_pca": bool(scale_before_pca),
        "n_samples": int(X.shape[0]),
        "state_counts": {s: int(np.sum(states_series.values == s)) for s in STATE_ORDER},
        "kruskal_wallis": global_kw,
        "directional_tests": directional_tests,
    }
    with open(traj_dir / "stats_pseudotime.json", "w", encoding="utf-8") as fh:
        json.dump(stats_obj, fh, indent=2, ensure_ascii=False)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Step4: DPT-based pseudotime rooted at S00.")
    p.add_argument("--matrix", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--k", type=int, default=15)
    p.add_argument("--n-dcs", type=int, default=20)
    p.add_argument("--random-state", type=int, default=42)
    p.add_argument(
        "--scale-before-pca",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether to z-score features with scanpy.pp.scale before PCA (default: enabled).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    run_step(
        matrix_path=args.matrix,
        output_dir=args.output_dir,
        k=args.k,
        n_dcs=args.n_dcs,
        random_state=args.random_state,
        scale_before_pca=args.scale_before_pca,
    )


if __name__ == "__main__":
    main()
