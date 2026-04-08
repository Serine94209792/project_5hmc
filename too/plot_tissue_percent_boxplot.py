#!/usr/bin/env python3
"""
Draw boxplot of tissue_percent.tsv: x-axis = tissue, y-axis = proportion.
With -m/--metadata: draw comparison boxplot (normal vs tumor per tissue).
Script and output path default to the same directory as this script (./too/).
"""
from __future__ import annotations

import argparse
import os

import pandas as pd
import numpy as np
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FLIERPROPS = dict(marker="o", markersize=2.5, alpha=0.7, linestyle="none")


def main() -> None:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    default_tsv = os.path.join(script_dir, "tissue_percent.tsv")
    default_metadata = os.path.join(project_root, "cfDNA_metadata2_TNM.csv")
    default_out = os.path.join(script_dir, "tissue_percent_boxplot.png")
    default_out_compare = os.path.join(script_dir, "tissue_percent_boxplot_normal_vs_tumor.png")

    parser = argparse.ArgumentParser(description="Boxplot of tissue proportions per sample.")
    parser.add_argument(
        "-i", "--input",
        default=default_tsv,
        help=f"Input TSV (default: {default_tsv})",
    )
    parser.add_argument(
        "-o", "--output",
        default=None,
        help="Output PNG (default: tissue_percent_boxplot.png or _normal_vs_tumor.png if -m)",
    )
    parser.add_argument(
        "-m", "--metadata",
        default=default_metadata,
        help="Metadata CSV with 'sample' and 'type' (0=normal, 1=tumor). If file exists, draw normal vs tumor comparison.",
    )
    args = parser.parse_args()

    df = pd.read_csv(args.input, sep="\t")
    # First column is sample ID (may be unnamed)
    id_col = df.columns[0]
    if id_col.startswith("Unnamed") or id_col == "":
        df = df.rename(columns={id_col: "sample"})
    else:
        df = df.rename(columns={id_col: "sample"})

    long = df.melt(
        id_vars=["sample"],
        var_name="tissue",
        value_name="proportion",
    )

    use_metadata = os.path.isfile(args.metadata)
    if args.metadata and not use_metadata:
        print(f"Warning: metadata file not found: {args.metadata}, drawing simple boxplot.")
    if use_metadata:
        meta = pd.read_csv(args.metadata)
        if "type" not in meta.columns or "sample" not in meta.columns:
            raise SystemExit("Metadata must contain columns 'sample' and 'type'.")
        meta = meta[["sample", "type"]].drop_duplicates()
        meta["group"] = meta["type"].map({0: "normal", 1: "tumor"})
        long = long.merge(meta[["sample", "group"]], on="sample", how="inner")
        if long.empty:
            raise SystemExit("No samples in common between tissue_percent and metadata.")
        out_path = args.output if args.output is not None else default_out_compare
        _plot_normal_vs_tumor(long, out_path)
    else:
        out_path = args.output if args.output is not None else default_out
        _plot_simple(long, out_path)
    print(f"Saved: {out_path}")


def _plot_simple(long: pd.DataFrame, output: str) -> None:
    fig, ax = plt.subplots(figsize=(14, 5))
    tissues = long["tissue"].unique()
    data_by_tissue = [long.loc[long["tissue"] == t, "proportion"].values for t in tissues]
    ax.boxplot(
        data_by_tissue,
        tick_labels=list(tissues),
        patch_artist=True,
        flierprops=FLIERPROPS,
        medianprops=dict(color="black"),
    )
    ax.set_xlabel("Tissue")
    ax.set_ylabel("Proportion")
    ax.set_title("Tissue proportion per sample")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(output, dpi=150)
    plt.close()


def _p_to_star(p: float) -> str:
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return ""


def _plot_normal_vs_tumor(long: pd.DataFrame, output: str) -> None:
    """Boxplot with two boxes per tissue: normal vs tumor; significance marked with *."""
    tissues = long["tissue"].unique()
    n_tissues = len(tissues)
    width = 0.35
    positions_n = [i - width / 2 for i in range(n_tissues)]
    positions_t = [i + width / 2 for i in range(n_tissues)]
    data_n = [long.loc[(long["tissue"] == t) & (long["group"] == "normal"), "proportion"].values for t in tissues]
    data_t = [long.loc[(long["tissue"] == t) & (long["group"] == "tumor"), "proportion"].values for t in tissues]

    # Significance: Mann-Whitney U per tissue
    stars = []
    y_max_per_tissue = []
    for i in range(n_tissues):
        a, b = data_n[i], data_t[i]
        a, b = a[~np.isnan(a)], b[~np.isnan(b)]
        y_max_per_tissue.append(max(np.max(a) if len(a) else 0, np.max(b) if len(b) else 0))
        if len(a) >= 2 and len(b) >= 2:
            try:
                _, p = stats.mannwhitneyu(a, b, alternative="two-sided")
                stars.append(_p_to_star(p))
            except Exception:
                stars.append("")
        else:
            stars.append("")

    fig, ax = plt.subplots(figsize=(14, 5))
    bp_n = ax.boxplot(
        data_n, positions=positions_n, widths=width * 0.8, patch_artist=True, flierprops=FLIERPROPS,
        medianprops=dict(color="black"),
    )
    bp_t = ax.boxplot(
        data_t, positions=positions_t, widths=width * 0.8, patch_artist=True, flierprops=FLIERPROPS,
        medianprops=dict(color="black"),
    )
    for patch in bp_n["boxes"]:
        patch.set_facecolor("lightblue")
    for patch in bp_t["boxes"]:
        patch.set_facecolor("lightcoral")
    ax.set_xticks(range(n_tissues))
    ax.set_xticklabels(tissues, rotation=45, ha="right")
    ax.set_xlabel("Tissue")
    ax.set_ylabel("Proportion")
    ax.set_title("Tissue proportion: normal vs tumor")
    ax.legend([bp_n["boxes"][0], bp_t["boxes"][0]], ["normal", "tumor"])

    # Draw significance asterisks above each tissue (keep inside figure)
    y_lo, y_hi = ax.get_ylim()
    y_range = y_hi - y_lo
    if any(stars):
        # Reserve top margin so asterisks stay inside; then place stars in lower part of margin
        y_top = y_hi + 0.15 * y_range
        ax.set_ylim(y_lo, y_top)
        for i in range(n_tissues):
            if stars[i]:
                # Place star at ~8% of margin above data, leaving room so text doesn't exceed y_top
                y_pos = y_max_per_tissue[i] + (y_top - y_max_per_tissue[i]) * 0.08
                ax.text(i, y_pos, stars[i], ha="center", va="bottom", fontsize=12)
    plt.tight_layout()
    plt.savefig(output, dpi=150)
    plt.close()


if __name__ == "__main__":
    main()
