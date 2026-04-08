import argparse
import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import seaborn as sns
from scipy.cluster import hierarchy
from scipy.stats import spearmanr
from sklearn.preprocessing import StandardScaler

# 图上字体调大
plt.rcParams["font.size"] = 14
plt.rcParams["axes.titlesize"] = 16
plt.rcParams["axes.labelsize"] = 15
plt.rcParams["xtick.labelsize"] = 14
plt.rcParams["ytick.labelsize"] = 14
plt.rcParams["legend.fontsize"] = 14


def transform_stage(stage_series: pd.Series) -> pd.Series:
    out = stage_series.copy()
    out = pd.to_numeric(out, errors="coerce")
    out = out.astype(object)
    out[out.isin([3, 4, 3.0, 4.0])] = "3/4"
    return out.astype(str)


def get_stage_color_map(stage_series: pd.Series):
    stage_color_dict = {
        "0": "#87CEEB", "0.0": "#87CEEB",
        "1": "#90EE90", "1.0": "#90EE90",
        "2": "#FFD700", "2.0": "#FFD700",
        "3": "#FF6347", "4": "#FF6347", "3/4": "#FF6347", "3.0": "#FF6347", "4.0": "#FF6347",
    }
    unique_stages = sorted(stage_series.astype(str).unique())
    return {s: stage_color_dict.get(s, sns.color_palette("husl", len(unique_stages))[unique_stages.index(s)]) for s in unique_stages}


def load_program_tsv(program_path: str) -> pd.DataFrame:
    return pd.read_csv(program_path, sep="\t")


def merge_W_by_program(base_dir: str, program_df: pd.DataFrame) -> pd.DataFrame:
    """
    按 program.tsv 选取各文件夹 nmf_W.tsv 的对应 component，列名改为 {dir}_C{n}，合并为一张表（行=样本，列=datatype_component）。
    program 列与文件夹名一致。
    """
    parts = []
    for _, row in program_df.iterrows():
        dirname, comp = row["program"], row["component"]
        w_path = os.path.join(base_dir, dirname, "nmf_W.tsv")
        w = pd.read_csv(w_path, sep="\t", index_col=0)
        col_name = f"{dirname}_{comp}"
        parts.append(w[[comp]].rename(columns={comp: col_name}))
    merged = pd.concat(parts, axis=1, join="inner")
    return merged


def plot_integrated_heatmap(W_df: pd.DataFrame, metadata: pd.DataFrame, output_path: str):
    meta = metadata.loc[metadata.index.isin(W_df.index)].copy()
    W_df = W_df.loc[meta.index]
    meta["stage"] = meta["stage"].astype(str)
    stage_order = sorted(meta["stage"].astype(str).unique())
    sample_order = []
    for s in stage_order:
        sample_order.extend(meta[meta["stage"] == s].index.tolist())
    W_sorted = W_df.loc[sample_order]
    M = W_sorted.T

    stage_to_color = get_stage_color_map(meta["stage"])
    col_colors = meta.loc[M.columns, "stage"].map(stage_to_color).fillna("#cccccc")
    # seaborn 要求 col_colors 为 list of arrays：每个元素为一“行”注释，长度=列数
    col_colors_list = [col_colors.values]

    row_linkage = hierarchy.linkage(M.values, method="average", metric="euclidean")
    g = sns.clustermap(
        M,
        row_linkage=row_linkage,
        col_cluster=False,
        cmap="viridis",
        vmin=-2,
        vmax=4,
        xticklabels=False,
        yticklabels=True,
        col_colors=col_colors_list,
        figsize=(12, max(6, M.shape[0] * 0.25)),
        cbar_kws={"label": "W (scaled)", "shrink": 0.8},
    )
    g.ax_heatmap.set_xlabel("Sample", fontsize=14)
    g.ax_heatmap.set_ylabel("Component", fontsize=14)
    g.ax_heatmap.tick_params(axis="y", labelsize=12)
    cbar_ax = getattr(g, "ax_cbar", None) or getattr(g, "cax", None)
    if cbar_ax is not None:
        cbar_ax.set_ylabel("W (scaled)", fontsize=13)
    # 添加 Stage 图例
    legend_handles = [
        Patch(facecolor=stage_to_color[s], label=f"Stage {s}")
        for s in stage_order
    ]
    g.ax_heatmap.legend(
        handles=legend_handles,
        title="Stage",
        loc="upper left",
        bbox_to_anchor=(1.02, 1),
        frameon=True,
        fontsize=11,
    )
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_integrated_violin(W_df: pd.DataFrame, metadata: pd.DataFrame, output_path: str):
    """小提琴图，显著性标在子图右侧。"""
    component_label_map = {
        "consensus_peak": "peak",
        "end_motif": "motif",
        "gene_counts": "gene",
    }

    def pretty_component_label(component_name: str) -> str:
        label = component_name
        for src, dst in component_label_map.items():
            label = label.replace(src, dst)
        return label

    meta = metadata.loc[metadata.index.isin(W_df.index)].copy()
    W_df = W_df.loc[meta.index]
    meta["stage"] = meta["stage"].astype(str)
    stage_order = sorted(meta["stage"].unique())

    rows = []
    for sample in W_df.index:
        stage = meta.loc[sample, "stage"]
        for col in W_df.columns:
            rows.append({"component": col, "stage": stage, "value": W_df.loc[sample, col]})
    long = pd.DataFrame(rows)

    comp_order = W_df.columns.tolist()
    n_comp = len(comp_order)
    fig, axes = plt.subplots(n_comp, 1, figsize=(8, max(4, n_comp * 1.4)), sharex=True)
    if n_comp == 1:
        axes = [axes]
    stage_to_color = get_stage_color_map(meta["stage"])
    stage_to_num = {s: i for i, s in enumerate(stage_order)}
    for i, comp in enumerate(comp_order):
        sub = long[long["component"] == comp]
        order = [s for s in stage_order if s in sub["stage"].unique()]
        palette = [stage_to_color.get(s, "#cccccc") for s in order]
        sns.violinplot(data=sub, x="stage", y="value", hue="stage", order=order, palette=palette, legend=False, ax=axes[i])
        axes[i].set_ylabel(pretty_component_label(comp), fontsize=15)
        axes[i].set_xlabel("")
        axes[i].tick_params(axis="both", labelsize=14)
        # 显著性标在图的右侧
        stage_num = sub["stage"].map(stage_to_num)
        valid = stage_num.notna() & sub["value"].notna()
        if valid.sum() >= 3:
            r, p = spearmanr(sub.loc[valid, "value"], stage_num.loc[valid])
            p_str = "p < 0.001" if p < 0.001 else f"p = {p:.3f}"
            txt = f"r = {r:.3f}\n{p_str}"
            axes[i].text(
                1.02, 0.5, txt,
                transform=axes[i].transAxes,
                ha="left", va="center",
                fontsize=14,
                family="sans-serif",
                bbox=dict(boxstyle="round,pad=0.35", facecolor="white", edgecolor="gray", alpha=0.6),
            )
    axes[-1].set_xlabel("Stage", fontsize=15)
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def build_feature_component_datatype_table(base_dir: str, program_df: pd.DataFrame) -> pd.DataFrame:
    """
    按 program.tsv 遍历所有 (program, component)，读各目录 H_top_features.tsv 与 nmf_H.tsv，
    得到长表：feature, component, datatype, H_loading。H_loading 为该 feature 在对应 component 的 H 原始值。
    """
    records = []
    for _, row in program_df.iterrows():
        dirname, comp = row["program"], row["component"]
        h_path = os.path.join(base_dir, dirname, "H_top_features.tsv")
        nmf_h_path = os.path.join(base_dir, dirname, "nmf_H.tsv")
        if not os.path.isfile(h_path) or not os.path.isfile(nmf_h_path):
            continue
        ht = pd.read_csv(h_path, sep="\t")
        ht = ht[ht["component"] == comp].copy()
        if ht.empty:
            continue
        ht["datatype"] = dirname
        nmf_h = pd.read_csv(nmf_h_path, sep="\t", index_col=0)
        if comp not in nmf_h.index:
            records.append(ht[["feature", "component", "datatype"]].assign(H_loading=float("nan")))
            continue
        row_h = nmf_h.loc[comp]
        ht["H_loading"] = ht["feature"].map(lambda f: row_h[f] if f in row_h.index else float("nan"))
        records.append(ht[["feature", "component", "datatype", "H_loading"]])
    if not records:
        return pd.DataFrame(columns=["feature", "component", "datatype", "H_loading"])
    return pd.concat(records, ignore_index=True).drop_duplicates(subset=["feature", "component", "datatype"])


def plot_integrated_H_top_features_count_bar(feat_table: pd.DataFrame, output_path: str):
    """
    堆叠条形图：横轴 = program 成分（按总 feature 数从高到低排序），纵轴 = top 特征数量。
    下段 = 独有（仅该 component），上段 = 重叠（同 datatype 其他 component 也有）；柱顶标注总数。
    """
    if feat_table.empty:
        return
    feat_table = feat_table.copy()
    feat_table["program_component"] = feat_table["datatype"] + "_" + feat_table["component"]

    # 1) 每个 (datatype, component) 的 feature 集合
    comp_sets = (
        feat_table.groupby(["datatype", "component"])["feature"]
        .apply(lambda x: set(x.dropna().unique()))
        .to_dict()
    )
    rows = []
    for (dt, comp), feat_set in comp_sets.items():
        other_feats = set()
        for (dt2, comp2), s in comp_sets.items():
            if dt2 == dt and comp2 != comp:
                other_feats |= s
        unique_set = feat_set - other_feats
        shared_set = feat_set & other_feats
        count_total = len(feat_set)
        count_unique = len(unique_set)
        count_shared = len(shared_set)
        rows.append({
            "datatype": dt,
            "component": comp,
            "program_component": f"{dt}_{comp}",
            "count": count_total,
            "count_unique": count_unique,
            "count_shared": count_shared,
        })
    count_per_comp = pd.DataFrame(rows)
    if count_per_comp.empty:
        return
    # 2) 按总数降序排序
    count_per_comp = count_per_comp.sort_values("count", ascending=False).reset_index(drop=True)
    comp_order = count_per_comp["program_component"].tolist()
    datatype_order = count_per_comp["datatype"].unique().tolist()

    fig, ax = plt.subplots(figsize=(max(8, len(comp_order) * 0.5), 6))
    pal = sns.color_palette("Set2", n_colors=8)
    if len(datatype_order) > len(pal):
        pal = sns.color_palette("tab10", n_colors=max(len(datatype_order), 10))
    datatype_palette = dict(zip(datatype_order, pal[: len(datatype_order)]))
    shared_color = "#cccccc"
    x = range(len(count_per_comp))
    # 下段：独有
    ax.bar(
        x,
        count_per_comp["count_unique"],
        bottom=0,
        color=[datatype_palette.get(dt, "#cccccc") for dt in count_per_comp["datatype"]],
        edgecolor="white",
        linewidth=0.8,
    )
    # 上段：重叠
    ax.bar(
        x,
        count_per_comp["count_shared"],
        bottom=count_per_comp["count_unique"],
        color=shared_color,
        edgecolor="white",
        linewidth=0.8,
    )
    y_max = count_per_comp["count"].max()
    ax.set_ylim(0, y_max * 1.08)
    # 柱顶标注 feature 总数（在 ylim 内留出空间，不越界）
    for i, total in enumerate(count_per_comp["count"]):
        ax.text(
            i,
            total,
            str(int(total)),
            ha="center",
            va="bottom",
            fontsize=11,
        )
    ax.set_xticks(x)
    ax.set_xticklabels(count_per_comp["program_component"], rotation=45, ha="right")
    ax.set_ylabel("Top feature count", fontsize=13)
    ax.set_xlabel("Program component", fontsize=13)
    ax.set_title("Integrated H: top feature count per program component", fontsize=14)
    legend_handles = [Patch(facecolor=datatype_palette[dt], label=dt) for dt in datatype_order]
    legend_handles.append(Patch(facecolor=shared_color, label="Shared"))
    ax.legend(handles=legend_handles, title="Datatype / Shared", fontsize=9, title_fontsize=10)
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def main(args):
    base_dir = args.base_dir
    out_dir = args.output_dir or base_dir
    os.makedirs(out_dir, exist_ok=True)

    metadata_path = args.metadata
    if not os.path.isabs(metadata_path) and not os.path.isfile(metadata_path):
        alt = os.path.normpath(os.path.join(base_dir, "..", metadata_path))
        if os.path.isfile(alt):
            metadata_path = alt
    if not os.path.isfile(metadata_path):
        raise FileNotFoundError(f"Metadata file not found: {args.metadata} (tried cwd and base_dir/..)")

    program_df = load_program_tsv(os.path.join(base_dir, "program.tsv"))
    W_merged = merge_W_by_program(base_dir, program_df)
    if W_merged.empty:
        raise SystemExit("No W columns merged. Check program.tsv and nmf_W.tsv under each folder.")
    W_merged.to_csv(os.path.join(out_dir, "integrated_W.tsv"), sep="\t")

    try:
        metadata = pd.read_csv(metadata_path, encoding="utf-8")
    except UnicodeDecodeError:
        metadata = pd.read_csv(metadata_path, encoding="latin1")
    # 去除列名首尾空格，避免 BOM 等导致列名不匹配
    metadata.columns = metadata.columns.str.strip()
    if "sample" not in metadata.columns or "stage" not in metadata.columns:
        raise ValueError("metadata must have columns 'sample' and 'stage'")
    metadata = metadata[metadata["stage"].notna()].copy()
    metadata["sample"] = metadata["sample"].astype(str).str.strip()
    metadata = metadata.drop_duplicates(subset=["sample"], keep="first")
    metadata.set_index("sample", inplace=True)
    metadata["stage"] = transform_stage(metadata["stage"])

    scaler = StandardScaler()
    W_scaled = pd.DataFrame(
        scaler.fit_transform(W_merged),
        index=W_merged.index,
        columns=W_merged.columns,
    )
    plot_integrated_heatmap(W_scaled, metadata, os.path.join(out_dir, "integrated_W_heatmap.png"))
    plot_integrated_violin(W_scaled, metadata, os.path.join(out_dir, "integrated_W_violin.png"))

    feat_table = build_feature_component_datatype_table(base_dir, program_df)
    feat_table.to_csv(os.path.join(out_dir, "integrated_H_top_features.tsv"), sep="\t")
    plot_integrated_H_top_features_count_bar(feat_table, os.path.join(out_dir, "integrated_H_top_features_count_bar.png"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Integrate W by program.tsv, plot, and build H top-feature matrix")
    parser.add_argument("--base_dir", required=True, help="tumor_likeness dir")
    parser.add_argument("-o", "--output_dir", default=None, help="Output directory (default: base_dir)")
    parser.add_argument(
        "-m", "--metadata",
        default="cfDNA_metadata2_TNM.csv",
        help="Metadata CSV with columns 'sample' and 'stage' (default: cfDNA_metadata2_TNM.csv)",
    )
    args = parser.parse_args()
    main(args)
