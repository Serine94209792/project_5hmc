import argparse
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Patch
from scipy.cluster import hierarchy
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import mutual_info_classif
from sklearn.decomposition import NMF
from statsmodels.stats.multitest import multipletests


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


def diff_features(
    X: pd.DataFrame,
    y: pd.Series,
    n_perm: int = 300,
    random_state: int = 0,
) -> pd.DataFrame:
    y_arr = y.astype(str).values
    feature_names = X.columns.tolist()
    n_features = len(feature_names)

    mi_obs = mutual_info_classif(X.values, y_arr, random_state=random_state, discrete_features=False, n_jobs=-1, n_neighbors=10)
    rng = np.random.default_rng(random_state)
    mi_perm = np.zeros((n_perm, n_features))
    for i in range(n_perm):
        y_perm = rng.permutation(y_arr)
        mi_perm[i] = mutual_info_classif(X.values, y_perm, random_state=random_state + 1 + i, discrete_features=False, n_jobs=-1, n_neighbors=10)
    mi_perm_std = mi_perm.std(axis=0)
    pvalue = (1 + (mi_perm >= mi_obs).sum(axis=0)) / (n_perm + 1)
    qvalue = multipletests(pvalue, method="fdr_bh")[1]
    results_df = pd.DataFrame(
        {"MI": mi_obs, "MI_std": mi_perm_std, "pvalue": pvalue, "qvalue": qvalue},
        index=feature_names,
    )
    return results_df


def run_nmf(
    diff_results: pd.DataFrame,
    df: pd.DataFrame,
    pvalue_threshold: float,
    n_components: int,
    random_state: int = 0,
):

    selected = diff_results.index[diff_results["pvalue"] < pvalue_threshold].tolist()
    selected = [f for f in selected if f in df.columns]
    if len(selected) == 0:
        raise ValueError(f"No feature with pvalue < {pvalue_threshold}")
    df_selected = df.loc[:, selected].copy()  # 非负矩阵
    col_mean = df_selected.mean(axis=0)
    col_mean = col_mean.replace(0, np.nan)
    df_selected = df_selected / col_mean
    df_selected = df_selected.fillna(1.0)
    nmf = NMF(
        n_components=n_components, 
        beta_loss="kullback-leibler", 
        solver="mu", 
        max_iter=1000,
        alpha_W=0.0,
        alpha_H=0.5,
        l1_ratio=1.0,
        random_state=random_state
    )
    W = nmf.fit_transform(df_selected.values)
    H = nmf.components_
    comp_names = [f"C{i+1}" for i in range(n_components)]
    W_df = pd.DataFrame(W, index=df_selected.index, columns=comp_names)
    H_df = pd.DataFrame(H, index=comp_names, columns=df_selected.columns)
    return W_df, H_df


def get_stage_color_map(stage_series: pd.Series):
    """与 fraction_visual / program_NMF 一致的 stage 颜色映射。"""
    stage_color_dict = {
        "0": "#87CEEB", "0.0": "#87CEEB",
        "1": "#90EE90", "1.0": "#90EE90",
        "2": "#FFD700", "2.0": "#FFD700",
        "3": "#FF6347", "4": "#FF6347", "3/4": "#FF6347", "3.0": "#FF6347", "4.0": "#FF6347",
    }
    unique_stages = sorted(stage_series.astype(str).unique())
    return {s: stage_color_dict.get(s, sns.color_palette("husl", len(unique_stages))[unique_stages.index(s)]) for s in unique_stages}


def plot_W_heatmap(W_df: pd.DataFrame, metadata: pd.DataFrame, output_path: str):
    """
    W 矩阵热图：行=components，列=样本；样本按 stage 升序；不显示列名，显示行名；行聚类；列侧注释 stage。
    """
    meta = metadata.loc[metadata.index.isin(W_df.index)].copy()
    W_df = W_df.loc[meta.index]
    stage_order = sorted(meta["stage"].astype(str).unique())
    sample_order = []
    for s in stage_order:
        sample_order.extend(meta[meta["stage"].astype(str) == s].index.tolist())
    W_sorted = W_df.loc[sample_order]
    M = W_sorted.T

    stage_to_color = get_stage_color_map(meta["stage"])
    col_colors = meta.loc[M.columns, "stage"].astype(str).map(stage_to_color).fillna("#cccccc")

    row_linkage = hierarchy.linkage(M.values, method="average", metric="euclidean")
    g = sns.clustermap(
        M,
        row_linkage=row_linkage,
        col_cluster=False,
        cmap="viridis",
        xticklabels=False,
        yticklabels=True,
        col_colors=col_colors.values,
        figsize=(12, 6),
        cbar_kws={"label": "W"},
    )
    g.ax_heatmap.set_xlabel("Sample")
    g.ax_heatmap.set_ylabel("Component")
    legend_handles = [Patch(facecolor=stage_to_color.get(s, "#cccccc"), label=f"Stage {s}") for s in stage_order]
    g.fig.legend(handles=legend_handles, title="Stage", bbox_to_anchor=(1.02, 0.5), loc="center left")
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_W_violin(W_df: pd.DataFrame, metadata: pd.DataFrame, output_path: str):
    """
    W 矩阵多行小提琴图：每行为一个 component，列为不同 stage（stage 升序），每个点为 W 矩阵元素。
    """
    meta = metadata.loc[metadata.index.isin(W_df.index)].copy()
    W_df = W_df.loc[meta.index]
    meta["stage"] = meta["stage"].astype(str)
    stage_order = sorted(meta["stage"].unique())

    rows = []
    for sample in W_df.index:
        stage = meta.loc[sample, "stage"]
        for comp in W_df.columns:
            rows.append({"component": comp, "stage": stage, "value": W_df.loc[sample, comp]})
    long = pd.DataFrame(rows)

    comp_order = W_df.columns.tolist()
    n_comp = len(comp_order)
    fig, axes = plt.subplots(n_comp, 1, figsize=(6, 6), sharex=True)
    if n_comp == 1:
        axes = [axes]
    stage_to_color = get_stage_color_map(meta["stage"])
    for i, comp in enumerate(comp_order):
        sub = long[long["component"] == comp]
        order = [s for s in stage_order if s in sub["stage"].unique()]
        palette = [stage_to_color.get(s, "#cccccc") for s in order]
        sns.violinplot(data=sub, x="stage", y="value", order=order, palette=palette, ax=axes[i])
        axes[i].set_ylabel(comp)
        axes[i].set_xlabel("")
    axes[-1].set_xlabel("Stage")
    plt.suptitle("W by component and stage", y=1.02)
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def transform_stage(stage_series: pd.Series) -> pd.Series:
    out = stage_series.copy()
    out = pd.to_numeric(out, errors="coerce")
    out = out.astype(object)
    out[out.isin([3, 4, 3.0, 4.0])] = "3/4"
    return out.astype(str)


def main(args):
    df = read_df(args.input_dir, args.datatype)
    # feature_analysis_matrix = read_feature_analysis_matrix(args.input_dir)
    
    # feature_analysis_matrix['abs_effect_mean'] = feature_analysis_matrix['effect_mean'].abs()
    # feature_analysis_matrix = feature_analysis_matrix.sort_values('abs_effect_mean', ascending=False)
    
    metadata = pd.read_csv(args.metadata)
    if "sample" not in metadata.columns:
        raise ValueError("metadata must contain 'sample' column")
    if "stage" not in metadata.columns:
        raise ValueError("metadata must contain stage column")
    
    metadata = metadata[metadata["stage"].notna()]
    metadata.set_index("sample", inplace=True)
    metadata["stage"] = transform_stage(metadata["stage"])
    # df = df.loc[metadata.index, feature_analysis_matrix.index]
    df = df.loc[metadata.index, :]
    scaler = StandardScaler()
    df_norm = scaler.fit_transform(df)
    df_norm = pd.DataFrame(df_norm, index=df.index, columns=df.columns)
    diff_results = diff_features(df_norm, metadata["stage"], n_perm=500)
    diff_results.to_csv(os.path.join(args.output_dir, "diff_results.tsv"), sep="\t", index=True)

    W_df, H_df = run_nmf(
        diff_results, df, args.pvalue_threshold, args.n_components
    )
    W_df.to_csv(os.path.join(args.output_dir, "nmf_W.tsv"), sep="\t", index=True)
    H_df.to_csv(os.path.join(args.output_dir, "nmf_H.tsv"), sep="\t", index=True)

    scaler = StandardScaler()
    W_df = pd.DataFrame(scaler.fit_transform(W_df), index=W_df.index, columns=W_df.columns)
    plot_W_heatmap(W_df, metadata, os.path.join(args.output_dir, "W_heatmap.png"))
    plot_W_violin(W_df, metadata, os.path.join(args.output_dir, "W_violin.png"))
    return W_df, H_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_dir", required=True, help="e.g. artemis/")
    parser.add_argument("-t", "--datatype", required=True, help="datatype", type=str,
                        choices=["artemis", "end_motif", "FSD", "gene_counts", "consensus_peak", "OCR", "window", "default"])
    parser.add_argument("-m", "--metadata", required=True, help="path to metadata file", type=str)
    parser.add_argument("-o", "--output_dir", required=True, help="path to output directory", type=str)
    parser.add_argument("-p", "--pvalue_threshold", type=float, default=0.05)
    parser.add_argument("-n", "--n_components", type=int, default=4)
    args = parser.parse_args()
    main(args)
