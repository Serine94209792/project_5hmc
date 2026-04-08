import pandas as pd
import os
import argparse
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt  # noqa: E402
import seaborn as sns  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402


def weighted_score(
    df: pd.DataFrame,
    feature_analysis_matrix: pd.DataFrame,
) -> pd.Series:
    """
    For each sample in df, compute a weighted score using features from feature_analysis_matrix:
    select those features, StandardScaler transform, multiply by effect_mean per feature,
    then sum and divide by number of features.
    """
    features = [f for f in feature_analysis_matrix.index if f in df.columns]
    X = df[features].astype(float)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    weights = feature_analysis_matrix.loc[features, "effect_mean"].values.astype(float)
    weighted = X_scaled * weights
    score = weighted.sum(axis=1) / len(features)
    return pd.Series(score, index=df.index)


def concordance(
    df: pd.DataFrame,
    feature_analysis_matrix: pd.DataFrame,
) -> pd.Series:
    """
    For each sample: scale the feature matrix, then for each feature compare the sign of
    effect_mean with the sign of that sample's scaled value for the feature. Return
    per-sample (number of concordant features) / (total features).
    """
    features = [f for f in feature_analysis_matrix.index if f in df.columns]
    if not features:
        return pd.Series(np.nan, index=df.index)
    X = df[features].astype(float)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    dir_effect = np.sign(feature_analysis_matrix.loc[features, "effect_mean"].values.astype(float))
    dir_scaled = np.sign(X_scaled)
    n_concordant = (dir_scaled == dir_effect).sum(axis=1)
    return pd.Series(n_concordant / len(features), index=df.index)


def concordance_and_weighted(
    df: pd.DataFrame,
    feature_analysis_matrix: pd.DataFrame,
) -> pd.DataFrame:
    """
    Combine concordance and weighted score for each sample. Returns a DataFrame
    with columns 'concordance' and 'weighted_score', index = sample index.
    """
    concordance_scores = concordance(df, feature_analysis_matrix)
    weighted_scores = weighted_score(df, feature_analysis_matrix)
    out = pd.concat([concordance_scores, weighted_scores], axis=1)
    out.columns = ["concordance", "weighted_score"]
    return out


def save_significant_regions_bed(
    sig_features: list,
    output_dir: str,
    output_filename: str = "significant_peaks.bed",
):
    """
    Save significant regions to BED file format

    Args:
        sig_features: List of region strings in format ["chr1_100_200", "chr2_300_400", ...]
        output_dir: Output directory to save the BED file
        output_filename: Name of the output BED file (default: "significant_peaks.bed")
    """
    if len(sig_features) == 0:
        print("Warning: No significant regions found. Creating empty BED file.")
        empty_bed = pd.DataFrame(columns=["chr", "start", "end"])
        empty_bed.to_csv(
            os.path.join(output_dir, output_filename),
            sep="\t",
            header=False,
            index=False
        )
        return

    # Parse region strings to BED format
    region_split = [r.split("_") for r in sig_features]
    if not all(len(r) == 3 for r in region_split):
        raise ValueError(f"Invalid region format. Expected 'chr_start_end', got: {sig_features[:5]}")
    
    bed_df = pd.DataFrame(region_split, columns=["chr", "start", "end"])
    bed_df["start"] = bed_df["start"].astype(int)
    bed_df["end"] = bed_df["end"].astype(int)
    
    # Sort by chr and start
    bed_df = bed_df.sort_values(["chr", "start", "end"])
    
    # Save to BED file (no header, tab-separated)
    bed_output_path = os.path.join(output_dir, output_filename)
    bed_df.to_csv(bed_output_path, sep="\t", header=False, index=False)
    print(f"Saved {len(bed_df)} significant regions to {bed_output_path}")
    return None


def read_df(
    input_dir: str,
    datatype: str,
):
    """
    Read dataframe and return dataframe with datatype columns

    Args:
        input_dir: str, input directory path
        datatype: str, datatype of the dataframe
    
    Returns:
        pd.DataFrame: 读取的数据框
    """
    if datatype == "artemis":
        df = pd.read_csv(os.path.join(input_dir, "artemis.tsv"), sep="\t", header=0, index_col=0)
    elif datatype == "end_motif":
        df = pd.read_csv(os.path.join(input_dir, "end_motif_matrix.tsv"), sep="\t", header=0, index_col=0)
    else:
        df = pd.read_csv(os.path.join(input_dir, "normalized_counts.tsv"), sep="\t", header=0, index_col=0)
        
    return df


def load_sample_list(path: str) -> set:
    with open(path) as f:
        return {line.strip() for line in f if line.strip()}


def build_sample_annotation(
    metadata: pd.DataFrame,
    annotation_path: str,
) -> pd.DataFrame:
    """
    Load type1 and type2 from sample_annotation.tsv and align to metadata samples.
    TSV must have columns: sample (or index), type1, type2.
    """
    ann = pd.read_csv(annotation_path, sep="\t", header=0, index_col=0)
    ann.index.name = "sample"
    if "type1" not in ann.columns or "type2" not in ann.columns:
        raise ValueError(
            f"sample_annotation.tsv must have columns type1 and type2; got {list(ann.columns)}"
        )
    ann = ann[["type1", "type2"]].reindex(metadata.index)
    return ann


def read_feature_analysis_matrix(
    input_dir: str,
    filename: str = "feature_analysis_matrix.tsv",
):
    """
    读取differential_analysis.py输出的summary_matrix
    
    Args:
        input_dir: str, 输入目录路径
        filename: str, summary_matrix文件名（default: "feature_analysis_matrix.tsv"）
    
    Returns:
        pd.DataFrame: summary_matrix，包含effect_mean, effect_CV, mean_log10qvalue列
    """
    summary_matrix_path = os.path.join(input_dir, filename)
    
    if not os.path.exists(summary_matrix_path):
        raise FileNotFoundError(f"Summary matrix file not found: {summary_matrix_path}")
    
    summary_matrix = pd.read_csv(summary_matrix_path, sep="\t", index_col=0)
    return summary_matrix


def get_annotation_palettes():
    """Return (type1_palette, type2_palette) for type1/type2 annotation colors."""
    type1 = {"tumor": "#C41E3A", "inflammation": "#FF7F50", "benign": "#2E8B57", "others": "#6A5ACD"}
    type2 = {"tumor": "#C41E3A", "predictable": "#3498DB", "abnormal": "#9B59B6"}
    return type1, type2


def plot_top_features_heatmap(
    df: pd.DataFrame,
    feature_analysis_matrix: pd.DataFrame,
    top_n: int,
    output_dir: str,
    metadata: pd.DataFrame,
    label: str,
    output_filename: str = "top_features_heatmap.png",
    annotation: pd.DataFrame | None = None,
):
    """
    Plot heatmap of top_n features with row/col clustering. If annotation is provided,
    use type1 and type2 for column color bars; otherwise use label (0/1).
    """
    top_features = feature_analysis_matrix.head(top_n).index.tolist()
    
    top_features_set = set(top_features)
    df_columns_set = set(df.columns)
    if not top_features_set.issubset(df_columns_set):
        missing = top_features_set - df_columns_set
        raise ValueError(f"Top features must be a subset of df.columns. {len(missing)} features not found in df: {list(missing)[:10]}")
    
    heatmap_data_raw = df[top_features]
    
    scaler = StandardScaler()
    heatmap_data_scaled = scaler.fit_transform(heatmap_data_raw)
    heatmap_data = pd.DataFrame(
        heatmap_data_scaled,
        index=heatmap_data_raw.index,
        columns=heatmap_data_raw.columns
    ).T
    
    n_features = len(top_features)
    fig_height = 8
    fig_width = 8

    sample_labels = metadata.loc[heatmap_data.columns, label]
    type_colors = sample_labels.map({"0": "lightblue", "1": "lightcoral"})
    if annotation is not None and heatmap_data.columns.isin(annotation.index).all():
        type1_pal, type2_pal = get_annotation_palettes()
        col_ann = annotation.loc[heatmap_data.columns]
        type1_colors = col_ann["type1"].map(type1_pal)
        type2_colors = col_ann["type2"].map(type2_pal)
        col_colors = pd.DataFrame({"type1": type1_colors, "type2": type2_colors, "type": type_colors})
    else:
        col_colors = pd.DataFrame({"type": type_colors})
    
    g = sns.clustermap(
        heatmap_data,
        row_cluster=True,
        col_cluster=True,
        cmap='RdBu_r',  # 红-白-蓝配色，反转后更美观
        center=0,  # 以0为中心
        figsize=(fig_width, fig_height),
        cbar_kws={'label': 'Expression Level'},
        xticklabels=False,
        yticklabels=False,
        method='ward',  # 使用ward方法进行聚类
        metric='euclidean',  # 使用欧氏距离
        col_colors=col_colors,  # 添加列颜色注释
    )
    
    # 使用figure对象强制设置大小
    g.fig.set_size_inches(fig_width, fig_height)
    g.fig.canvas.draw()  # 强制重绘以确保大小生效
    
    g.fig.suptitle(f'Top {n_features} Features Heatmap', y=1.02, fontsize=14, fontweight='bold')
    heatmap_path = os.path.join(output_dir, output_filename)
    # 不使用bbox_inches='tight'以保持指定的图形大小
    g.savefig(heatmap_path, dpi=300, bbox_inches=None, pad_inches=0.1)
    plt.close()
    print(f"Saved top {n_features} features heatmap to {heatmap_path}")
    
    return None


def plot_top_features_violin(
    df: pd.DataFrame,
    feature_analysis_matrix: pd.DataFrame,
    top_n: int,
    output_dir: str,
    metadata: pd.DataFrame,
    label: str,
    output_filename: str = "top_features_violin.png",
):
    """
    绘制前top_n个特征的小提琴图，对比0和1组
    
    Args:
        df: pd.DataFrame, (n_samples, n_features) 数据矩阵
        feature_analysis_matrix: pd.DataFrame, 特征分析矩阵，index为特征名
        top_n: int, 要绘制的特征数量
        output_dir: str, 输出目录
        metadata: pd.DataFrame, 元数据，包含label列
        label: str, label列名
        output_filename: str, 输出文件名（default: "top_features_violin.png"）
    """
    top_features = feature_analysis_matrix.head(top_n).index.tolist()
    
    top_features_set = set(top_features)
    df_columns_set = set(df.columns)
    if not top_features_set.issubset(df_columns_set):
        missing = top_features_set - df_columns_set
        raise ValueError(f"Top features must be a subset of df.columns. {len(missing)} features not found in df: {list(missing)[:10]}")
    
    plot_data = df[top_features].copy()  # 确保使用原始数据，不进行标准化
    # 检查数据是否有负值（用于调试）
    if (plot_data < 0).any().any():
        print(f"Warning: Found negative values in plot_data. Min value: {plot_data.min().min()}")
    sample_labels = metadata.loc[plot_data.index, label]
    plot_data_with_label = plot_data.copy()
    plot_data_with_label['label'] = sample_labels
    
    # 计算图形大小
    n_features = len(top_features)
    n_cols = min(3, n_features)  # 每行最多3个图
    n_rows = (n_features + n_cols - 1) // n_cols
    fig_width = n_cols * 3
    fig_height = n_rows * 3
    
    # 创建图形
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_width, fig_height))
    if n_features == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    # 为每个特征绘制小提琴图
    for idx, feature in enumerate(top_features):
        ax = axes[idx]
        
        # 对比0和1组
        group_0_data = plot_data_with_label[plot_data_with_label['label'] == "0"][feature].values
        group_1_data = plot_data_with_label[plot_data_with_label['label'] == "1"][feature].values
        
        # 绘制小提琴图
        parts = ax.violinplot([group_0_data, group_1_data], positions=[0, 1], showmeans=True, showmedians=True)
        
        # 美化小提琴图
        colors = ['lightblue', 'lightcoral']
        for i, pc in enumerate(parts['bodies']):
            pc.set_facecolor(colors[i])
            pc.set_alpha(0.7)
        
        parts['cmeans'].set_color('red')
        parts['cmeans'].set_linewidth(2)
        parts['cmedians'].set_color('green')
        parts['cmedians'].set_linewidth(2)
        
        # 添加散点图显示数据点
        y_jitter_0 = np.random.normal(0, 0.02, len(group_0_data))
        y_jitter_1 = np.random.normal(0, 0.02, len(group_1_data))
        ax.scatter([0] * len(group_0_data), group_0_data + y_jitter_0, alpha=0.3, s=10, color='black')
        ax.scatter([1] * len(group_1_data), group_1_data + y_jitter_1, alpha=0.3, s=10, color='black')
        
        # 设置标签
        ax.set_xticks([0, 1])
        ax.set_xticklabels(['0', '1'], fontsize=10)
        
        ax.set_ylabel('Value', fontsize=10)
        ax.set_title(f'{feature}', fontsize=10, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
    
    # 隐藏多余的子图
    for idx in range(n_features, len(axes)):
        axes[idx].axis('off')
    
    plt.suptitle(f'Top {n_features} Features Violin Plot', fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    
    # 保存图片
    violin_path = os.path.join(output_dir, output_filename)
    plt.savefig(violin_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved top {n_features} features violin plot to {violin_path}")
    
    return None


def plot_pca(
    df: pd.DataFrame,
    feature_analysis_matrix: pd.DataFrame,
    top_n: int,
    metadata: pd.DataFrame,
    label: str,
    output_dir: str,
    output_filename: str = "pca_plot.png",
    annotation: pd.DataFrame | None = None,
):
    """
    PCA plot (top_n features, first two PCs). If annotation is provided, draw two subplots
    colored by type1 and type2; otherwise one plot by label 0/1.
    """
    top_features = feature_analysis_matrix.head(top_n).index.tolist()
    top_features_set = set(top_features)
    df_columns_set = set(df.columns)
    if not top_features_set.issubset(df_columns_set):
        missing = top_features_set - df_columns_set
        raise ValueError(f"Top features must be a subset of df.columns. {len(missing)} features not found in df: {list(missing)[:10]}")
    
    df_top = df[top_features]
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df_top)
    df_scaled = pd.DataFrame(df_scaled, index=df.index, columns=df_top.columns)
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(df_scaled)
    pca_df = pd.DataFrame(pca_result, index=df.index, columns=['PC1', 'PC2'])
    explained_var = pca.explained_variance_ratio_

    sample_labels = metadata.loc[pca_df.index, label]
    type_pal = {"0": "lightblue", "1": "lightcoral"}
    if annotation is not None and pca_df.index.isin(annotation.index).all():
        pca_df = pca_df.join(annotation[["type1", "type2"]])
        pca_df["type"] = sample_labels
        type1_pal, type2_pal = get_annotation_palettes()
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 6))
        for ax, col, pal in [
            (ax1, "type1", type1_pal),
            (ax2, "type2", type2_pal),
            (ax3, "type", type_pal),
        ]:
            for cat in pca_df[col].dropna().unique():
                sub = pca_df[pca_df[col] == cat]
                ax.scatter(
                    sub["PC1"], sub["PC2"],
                    c=pal.get(cat, "#333333"),
                    label=cat,
                    alpha=0.7, s=30, edgecolors="black", linewidths=0.5,
                )
            ax.set_xlabel(f'PC1 ({explained_var[0]*100:.2f}% variance)', fontsize=10, fontweight='bold')
            ax.set_ylabel(f'PC2 ({explained_var[1]*100:.2f}% variance)', fontsize=10, fontweight='bold')
            ax.set_title(f'PCA by {col}', fontsize=11, fontweight='bold')
            ax.legend(fontsize=8, loc='best', framealpha=0.9)
            ax.grid(True, alpha=0.3)
            ax.set_axisbelow(True)
        plt.tight_layout()
    else:
        pca_df["type"] = sample_labels
        fig, ax = plt.subplots(figsize=(6, 6))
        for label_val in ["0", "1"]:
            label_data = pca_df[pca_df["type"] == label_val]
            ax.scatter(
                label_data["PC1"], label_data["PC2"],
                c=type_pal[label_val],
                label=f"Group {label_val}",
                alpha=0.7, s=30, edgecolors="black", linewidths=0.5,
            )
        ax.set_xlabel(f'PC1 ({explained_var[0]*100:.2f}% variance)', fontsize=12, fontweight='bold')
        ax.set_ylabel(f'PC2 ({explained_var[1]*100:.2f}% variance)', fontsize=12, fontweight='bold')
        ax.set_title(f'PCA Plot (Top {top_n} Features)', fontsize=14, fontweight='bold')
        ax.legend(fontsize=11, loc='best', framealpha=0.9)
        ax.grid(True, alpha=0.3)
        ax.set_axisbelow(True)
        plt.tight_layout()

    pca_path = os.path.join(output_dir, output_filename)
    plt.savefig(pca_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved PCA plot to {pca_path}")
    print(f"PC1 explains {explained_var[0]*100:.2f}% of variance, PC2 {explained_var[1]*100:.2f}%")
    return None


def plot_concordance_weighted_scatter(
    score_df: pd.DataFrame,
    annotation: pd.DataFrame,
    metadata: pd.DataFrame,
    label: str,
    output_dir: str,
    output_filename: str = "concordance_weighted_scatter.png",
) -> None:
    """
    Scatter: x=weighted_score, y=concordance. Three panels colored by type1, type2, type.
    Style aligned with plot_pca (palettes, alpha, s, edgecolors).
    """
    if "concordance" not in score_df.columns or "weighted_score" not in score_df.columns:
        raise ValueError("score_df must have columns 'concordance' and 'weighted_score'")
    plot_df = score_df[["concordance", "weighted_score"]].copy()
    common = plot_df.index.intersection(annotation.index)
    if len(common) == 0:
        raise ValueError("score_df index and annotation index have no overlap")
    plot_df = plot_df.loc[common].join(annotation.loc[common, ["type1", "type2"]])
    plot_df["type"] = metadata.loc[common, label].astype(str)
    type1_pal, type2_pal = get_annotation_palettes()
    type_pal = {"0": "lightblue", "1": "lightcoral"}
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 6))
    for ax, col, pal in [
        (ax1, "type1", type1_pal),
        (ax2, "type2", type2_pal),
        (ax3, "type", type_pal),
    ]:
        for cat in plot_df[col].dropna().unique():
            sub = plot_df[plot_df[col] == cat]
            ax.scatter(
                sub["weighted_score"],
                sub["concordance"],
                c=pal.get(cat, "#333333"),
                label=cat,
                alpha=0.7,
                s=30,
                edgecolors="black",
                linewidths=0.5,
            )
        ax.set_xlabel("weighted_score", fontsize=10, fontweight="bold")
        ax.set_ylabel("concordance", fontsize=10, fontweight="bold")
        ax.set_title(f"By {col}", fontsize=11, fontweight="bold")
        ax.legend(fontsize=8, loc="best", framealpha=0.9)
        ax.grid(True, alpha=0.3)
        ax.set_axisbelow(True)
    plt.suptitle("Concordance vs weighted score", fontsize=12, fontweight="bold", y=1.02)
    plt.tight_layout()
    out_path = os.path.join(output_dir, output_filename)
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved concordance vs weighted_score scatter to {out_path}")
    return None


def plot_normal_pca(
    df: pd.DataFrame,
    feature_analysis_matrix: pd.DataFrame,
    top_n: int,
    metadata: pd.DataFrame,
    label: str,
    output_dir: str,
    output_filename: str = "pca_normal_plot.png",
    annotation: pd.DataFrame | None = None,
) -> None:
    """
    PCA on type==0 samples only (top_n features, first two PCs). Annotation uses
    only type1 and type2 (two subplots). Style same as plot_pca.
    """
    type0_mask = metadata.loc[df.index, label] == "0"
    df_normal = df.loc[type0_mask]
    if len(df_normal) == 0:
        print("Warning: No type==0 samples, skipping plot_normal_pca")
        return None
    top_features = feature_analysis_matrix.head(top_n).index.tolist()
    top_features_set = set(top_features)
    df_columns_set = set(df_normal.columns)
    if not top_features_set.issubset(df_columns_set):
        missing = top_features_set - df_columns_set
        raise ValueError(f"Top features must be a subset of df.columns. {len(missing)} features not found in df: {list(missing)[:10]}")
    df_top = df_normal[top_features]
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df_top)
    df_scaled = pd.DataFrame(df_scaled, index=df_normal.index, columns=df_top.columns)
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(df_scaled)
    pca_df = pd.DataFrame(pca_result, index=df_normal.index, columns=["PC1", "PC2"])
    explained_var = pca.explained_variance_ratio_
    if annotation is not None and pca_df.index.isin(annotation.index).all():
        pca_df = pca_df.join(annotation.loc[pca_df.index, ["type1", "type2"]])
        type1_pal, type2_pal = get_annotation_palettes()
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
        for ax, col, pal in [
            (ax1, "type1", type1_pal),
            (ax2, "type2", type2_pal),
        ]:
            for cat in pca_df[col].dropna().unique():
                sub = pca_df[pca_df[col] == cat]
                ax.scatter(
                    sub["PC1"], sub["PC2"],
                    c=pal.get(cat, "#333333"),
                    label=cat,
                    alpha=0.7, s=30, edgecolors="black", linewidths=0.5,
                )
            ax.set_xlabel(f"PC1 ({explained_var[0]*100:.2f}% variance)", fontsize=10, fontweight="bold")
            ax.set_ylabel(f"PC2 ({explained_var[1]*100:.2f}% variance)", fontsize=10, fontweight="bold")
            ax.set_title(f"PCA by {col} (type==0 only)", fontsize=11, fontweight="bold")
            ax.legend(fontsize=8, loc="best", framealpha=0.9)
            ax.grid(True, alpha=0.3)
            ax.set_axisbelow(True)
        plt.suptitle(f"PCA (type==0, n={len(pca_df)}, top {top_n} features)", fontsize=12, fontweight="bold", y=1.02)
        plt.tight_layout()
    else:
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.scatter(
            pca_df["PC1"], pca_df["PC2"],
            c="lightblue", label="type=0",
            alpha=0.7, s=30, edgecolors="black", linewidths=0.5,
        )
        ax.set_xlabel(f"PC1 ({explained_var[0]*100:.2f}% variance)", fontsize=12, fontweight="bold")
        ax.set_ylabel(f"PC2 ({explained_var[1]*100:.2f}% variance)", fontsize=12, fontweight="bold")
        ax.set_title(f"PCA (type==0 only, top {top_n} features)", fontsize=14, fontweight="bold")
        ax.legend(fontsize=11, loc="best", framealpha=0.9)
        ax.grid(True, alpha=0.3)
        ax.set_axisbelow(True)
        plt.tight_layout()
    pca_path = os.path.join(output_dir, output_filename)
    plt.savefig(pca_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved normal PCA plot (type==0) to {pca_path}")
    return None


def main(args):
    df = read_df(args.input_dir, args.datatype)
    feature_analysis_matrix = read_feature_analysis_matrix(args.input_dir)
    
    feature_analysis_matrix['abs_effect_mean'] = feature_analysis_matrix['effect_mean'].abs()
    feature_analysis_matrix = feature_analysis_matrix.sort_values('abs_effect_mean', ascending=False)
    
    metadata = pd.read_csv(args.metadata)
    if "sample" not in metadata.columns:
        raise ValueError("metadata must contain 'sample' column")
    if args.label not in metadata.columns:
        raise ValueError(f"metadata must contain '{args.label}' column")
    
    metadata = metadata[metadata[args.label].notna()]
    metadata.set_index("sample", inplace=True)
    metadata[args.label] = metadata[args.label].astype(int).astype(str)
    
    df_samples_set = set(df.index)
    metadata_samples_set = set(metadata.index)
    
    if df_samples_set != metadata_samples_set:
        missing_in_df = metadata_samples_set - df_samples_set
        missing_in_metadata = df_samples_set - metadata_samples_set
        error_msg = "df.index and metadata.index must be identical. "
        if len(missing_in_df) > 0:
            error_msg += f"{len(missing_in_df)} samples in metadata but not in df: {list(missing_in_df)[:10]}. "
        if len(missing_in_metadata) > 0:
            error_msg += f"{len(missing_in_metadata)} samples in df but not in metadata: {list(missing_in_metadata)[:10]}."
        raise ValueError(error_msg)

    fs_dir = os.path.dirname(os.path.abspath(__file__))
    annotation_path = os.path.join(fs_dir, "sample_annotation.tsv")
    if not os.path.exists(annotation_path):
        raise FileNotFoundError(
            f"Sample annotation not found: {annotation_path}. "
            "Please provide feature_selection/sample_annotation.tsv with columns: sample, type1, type2."
        )
    annotation = build_sample_annotation(metadata, annotation_path)
    ann_path = os.path.join(args.output_dir, "sample_annotation.tsv")
    annotation.reset_index().to_csv(ann_path, sep="\t", index=False)
    print(f"Saved sample annotation (sample, type1, type2) to {ann_path}")
    
    plot_top_features_heatmap(
        df=df,
        feature_analysis_matrix=feature_analysis_matrix,
        top_n=args.heatmap_top_n,
        output_dir=args.output_dir,
        metadata=metadata,
        label=args.label,
        annotation=annotation,
    )
    
    plot_top_features_violin(
        df=df,
        feature_analysis_matrix=feature_analysis_matrix,
        top_n=args.violin_top_n,
        output_dir=args.output_dir,
        metadata=metadata,
        label=args.label,
    )
    
    plot_pca(
        df=df,
        feature_analysis_matrix=feature_analysis_matrix,
        top_n=args.heatmap_top_n,
        metadata=metadata,
        label=args.label,
        output_dir=args.output_dir,
        annotation=annotation,
    )
    plot_normal_pca(
        df=df,
        feature_analysis_matrix=feature_analysis_matrix,
        top_n=args.heatmap_top_n,
        metadata=metadata,
        label=args.label,
        output_dir=args.output_dir,
        annotation=annotation,
    )

    score_df = concordance_and_weighted(df, feature_analysis_matrix)
    concordance_tsv_path = os.path.join(args.output_dir, "concordance_weighted.tsv")
    score_df.to_csv(concordance_tsv_path, sep="\t", index=True)
    print(f"Saved concordance and weighted score matrix to {concordance_tsv_path}")
    plot_concordance_weighted_scatter(
        score_df=score_df,
        annotation=annotation,
        metadata=metadata,
        label=args.label,
        output_dir=args.output_dir,
    )
    
    if args.datatype in ["consensus_peak", "OCR", "window"]:
        all_features = feature_analysis_matrix.index.tolist()
        bed_filename_map = {
            "consensus_peak": "significant_peaks.bed",
            "OCR": "significant_OCR.bed",
            "window": "significant_windows.bed",
        }
        bed_filename = bed_filename_map[args.datatype]
        
        save_significant_regions_bed(
            sig_features=all_features,
            output_dir=args.output_dir,
            output_filename=bed_filename,
        )
    
    return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate differential features")
    parser.add_argument("-i", "--input_dir", required=True, help="e.g. artemis/")
    parser.add_argument("-t", "--datatype", required=True, help="datatype", type=str,
                        choices=["artemis", "end_motif", "FSD", "gene_counts", "consensus_peak", "OCR", "window", "default"])
    parser.add_argument("-m", "--metadata", required=True, help="path to metadata file", type=str)
    parser.add_argument("-l", "--label", required=True, help="label column name in metadata", type=str)
    parser.add_argument("-o", "--output_dir", default="./", help="output directory path", type=str)
    parser.add_argument("-hn", "--heatmap_top_n", type=int, default=50, help="number of top features for heatmap (default: 50)")
    parser.add_argument("-vn", "--violin_top_n", type=int, default=6, help="number of top features for violin plot (default: 20)")
    args = parser.parse_args()
    main(args)