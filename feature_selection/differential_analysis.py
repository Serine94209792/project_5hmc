import pandas as pd
import os
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend to avoid GUI issues
import matplotlib.pyplot as plt  # noqa: E402
from pydeseq2.dds import DeseqDataSet  # noqa: E402
from pydeseq2.default_inference import DefaultInference  # noqa: E402
from pydeseq2.ds import DeseqStats  # noqa: E402
import argparse  # noqa: E402
import numpy as np  # noqa: E402
from statsmodels.stats.multitest import multipletests  # noqa: E402
import patsy  # noqa: E402
from inmoose.limma import lmFit, contrasts_fit, eBayes  # noqa: E402
from statsmodels.othermod.betareg import BetaModel  # noqa: E402
import statsmodels.api as sm  # noqa: E402
from sklearn.model_selection import RepeatedKFold  # noqa: E402
# import seaborn as sns  # noqa: E402


def transform_counts(
    df: pd.DataFrame,
    metadata: pd.DataFrame,
    label: str = "type",
    n_cpus: int = 8,
):
    """
    df has to be (sample, feature)
    
    Args:
        df: pd.DataFrame, (n_samples, n_features)
        metadata: pd.DataFrame, 元数据
        label: str, label列名
        n_cpus: int, 使用的CPU核心数 (default: 1)
    """
    dds = DeseqDataSet(
        counts=df,
        metadata=metadata,
        design=label,
        refit_cooks=False,
        n_cpus=n_cpus,
    )
    dds.deseq2()

    # Get normalized counts and convert to DataFrame
    # dds.layers["normed_counts"] is (n_samples, n_peaks) numpy array
    normalized_counts_array = dds.layers["normed_counts"]
    normalized_counts = pd.DataFrame(
        normalized_counts_array,
        index=df.index,
        columns=df.columns
    )

    dds.vst(use_design=True)
    # Get VST counts and convert to DataFrame
    # dds.layers["vst_counts"] is (n_samples, n_peaks) numpy array
    vst_counts_array = dds.layers["vst_counts"]
    vst_counts = pd.DataFrame(
        vst_counts_array,
        index=df.index,
        columns=df.columns
    )
    return normalized_counts, vst_counts


def differential_analysis_counts(
    df: pd.DataFrame,
    metadata: pd.DataFrame,
    output_dir: str,
    logfc_threshold: float,
    label: str = "type",
    qvalue_threshold: float = 0.05,
    n_cpus: int = 8,
):
    """
    df: pd.DataFrame, (n_samples, n_features)
    df has to be raw counts
    
    Args:
        df: pd.DataFrame, (n_samples, n_features)
        metadata: pd.DataFrame, 元数据
        output_dir: str, 输出目录
        logfc_threshold: float, logfc阈值
        label: str, label列名
        qvalue_threshold: float, qvalue阈值
        n_cpus: int, 使用的CPU核心数 (default: 1)
    """
    inference = DefaultInference(n_cpus=n_cpus)
    dds = DeseqDataSet(
        counts=df,
        metadata=metadata,
        design=label,
        refit_cooks=False,
        n_cpus=n_cpus,
    )
    dds.deseq2()
    stats = DeseqStats(
        dds,
        alpha=0.05,
        contrast=[label, "1", "0"],  # 1 vs 0: logFC > 0 means 1 > 0 (experimental > control)
        inference=inference,
        cooks_filter=False,
        independent_filter=True,
    )
    stats.summary()
    results_df = stats.results_df
    results_df = results_df.sort_values(by="padj", ascending=True)
    results_df["-log10qvalue"] = -np.log10(results_df["padj"].clip(lower=1e-300))
    results_df["-log10qvalue"] = results_df["-log10qvalue"].clip(upper=300)
    results_df["Significant"] = (
        (results_df["padj"] < qvalue_threshold) &
        (results_df["log2FoldChange"].abs() >= logfc_threshold)
    )
    
    sig_features = results_df[results_df["Significant"]].index.tolist()
    sig_results_df = results_df[results_df["Significant"]].copy()
    sig_results_df.to_csv(os.path.join(output_dir, "diff_results.tsv"), sep="\t", index=True)

    return sig_features


def differential_analysis_tpm(
    df: pd.DataFrame,
    metadata: pd.DataFrame,
    output_dir: str,
    logfc_threshold: float,
    label: str = "type",
    qvalue_threshold: float = 0.05,
):
    """
    df: pd.DataFrame, (n_samples, n_features)
    df is a log1p(tpm) matrix
    """
    design = patsy.dmatrix(
        f"{label}",
        {
            label: metadata[label]
        }
    )
    # limma input df has to be (n_features, n_samples)
    df = df.T
    fit = lmFit(df, design)

    # Get the column name for label[T.1] from design matrix
    design_cols = design.design_info.column_names
    label_col = None
    for col in design_cols:
        if f'{label}[T.1]' in col or (label in col and 'T.1' in col):
            label_col = col
            break

    if label_col is None:
        raise ValueError(f"Could not find column for {label}[T.1] in design matrix. Available columns: {design_cols}")

    # Directly construct contrast matrix to avoid makeContrasts eval issues
    # Create a contrast vector: 1 for label[T.1], 0 for others
    contrast_vec = np.zeros(len(design_cols))
    label_idx = design_cols.index(label_col)
    contrast_vec[label_idx] = 1.0
    ctr_matrix = pd.DataFrame(
        contrast_vec.reshape(-1, 1),
        index=design_cols,
        columns=[f"{label}_1_vs_0"]
    )
    fit2 = contrasts_fit(fit, ctr_matrix)
    fit_eb = eBayes(fit2)
    contrast_name = f"{label}_1_vs_0"
    feature_names = df.index.tolist()

    # 从 fit_eb 对象中提取数据
    # coefficients 对应 logFC (log2FoldChange)
    # t 对应 t 统计量
    # p_value 对应 p 值
    # lods 对应 B 统计量 (lods)
    # F 和 F_p_value 是整体 F 统计量
    coefficients = np.array(fit_eb.coefficients[contrast_name]).flatten()
    t_stats = np.array(fit_eb.t[contrast_name]).flatten()
    p_values = np.array(fit_eb.p_value[contrast_name]).flatten()
    lods = np.array(fit_eb.lods[contrast_name]).flatten()
    F_stats = np.array(fit_eb.F).flatten()
    F_p_values = np.array(fit_eb.F_p_value).flatten()
    baseMean = np.array(fit_eb.Amean).flatten()
    _, qvalues, _, _ = multipletests(p_values, method="fdr_bh", alpha=0.05)

    result = pd.DataFrame({
        "log2FoldChange": coefficients,
        "t": t_stats,
        "pvalue": p_values,
        "qvalue": qvalues,
        "lods": lods,
        "F": F_stats,
        "F_p_value": F_p_values,
        "baseMean": baseMean,
    }, index=feature_names)

    result = result.sort_values(by="qvalue", ascending=True)
    result["-log10qvalue"] = -np.log10(result["qvalue"].clip(lower=1e-300))
    result["-log10qvalue"] = result["-log10qvalue"].clip(upper=300)
    result["Significant"] = (
        (result["qvalue"] < qvalue_threshold) &
        (result["log2FoldChange"].abs() >= logfc_threshold)
    )
    
    sig_features = result[result["Significant"]].index.tolist()
    sig_results_df = result[result["Significant"]].copy()
    sig_results_df.to_csv(os.path.join(output_dir, "diff_results.tsv"), sep="\t", index=True)

    return sig_features


def differential_analysis_prop(
    df: pd.DataFrame,
    metadata: pd.DataFrame,
    output_dir: str,
    label: str = "type",
    coefficient_threshold: float = 0.1,
    qvalue_threshold: float = 0.05,
):
    """
    df has to be (n_samples, n_features)
    df is a proportion matrix, elements are between 0 and 1
    """
    design_data = metadata.loc[df.index, [label]]

    feature_names = df.columns.tolist()
    n_features = len(feature_names)

    coefficients = np.zeros(n_features)
    pvalues = np.ones(n_features)
    tvalues = np.zeros(n_features)
    se = np.zeros(n_features)

    formula = f"{label}"
    design_matrix = patsy.dmatrix(formula, design_data, return_type='dataframe')

    # Get the column name for the label contrast
    # Find the column that represents the contrast (e.g., label[T.1] for label="0" vs "1")
    design_cols = design_matrix.columns.tolist()
    label_col = None
    for col in design_cols:
        if f'{label}[T.' in col or (label in col and 'T.' in col):
            label_col = col
            break

    if label_col is None:
        raise ValueError(f"Could not find contrast column for {label} in design matrix")

    # Fit beta regression for each feature
    for idx, feature in enumerate(feature_names):
        if (idx + 1) % 100 == 0:
            print(f"Processing feature {idx + 1}/{n_features}...")

        y = df[feature].values
        model = BetaModel(y, design_matrix, link=sm.families.links.logit())
        result = model.fit()
        coefficients[idx] = result.params[label_col]
        tvalues[idx] = result.tvalues[label_col]
        pvalues[idx] = result.pvalues[label_col]
        se[idx] = result.bse[label_col]

    _, qvalues, _, _ = multipletests(pvalues, method='fdr_bh', alpha=0.05)
    results_df = pd.DataFrame({
        'feature': feature_names,
        'coefficient': coefficients,
        'std_error': se,
        't': tvalues,
        'pvalue': pvalues,
        'qvalue': qvalues
    })

    results_df = results_df.sort_values('qvalue', ascending=True)
    results_df['-log10qvalue'] = -np.log10(results_df['qvalue'].clip(lower=1e-300))
    results_df['-log10qvalue'] = results_df['-log10qvalue'].clip(upper=300)
    results_df['Significant'] = (
        (results_df['qvalue'] < qvalue_threshold) &
        (results_df['coefficient'].abs() >= coefficient_threshold)
    )
    
    sig_features = results_df[results_df['Significant']]['feature'].tolist()
    sig_results_df = results_df[results_df['Significant']].copy()
    output_path = os.path.join(output_dir, "diff_results.tsv")
    sig_results_df.to_csv(output_path, sep="\t", index=False)

    return sig_features


def split_exp_set(
    metadata: pd.DataFrame,
    label: str = "type",
    n_folds: int = 3,
    n_repeats: int = 3,
    random_state: int = 42,
):
    """
    将label=1的样本分成n_folds折，重复n_repeats次
    每一折与所有label=0的样本构成新的dataset
    返回所有dataset对应的样本列表
    
    Args:
        metadata: pd.DataFrame, 包含label列的元数据
        label: str, label列名
        n_folds: int, 将1类分成几折 (default: 3)
        n_repeats: int, 重复几次 (default: 3)
        random_state: int, 随机种子 (default: 42)
    
    Returns:
        list: 包含n_folds * n_repeats个样本列表的列表，每个元素是一个dataset的样本列表
    """
    samples_1 = metadata[metadata[label] == "1"].index.tolist()
    samples_0 = metadata[metadata[label] == "0"].index.tolist()
    
    if len(samples_1) == 0:
        raise ValueError(f"No samples with {label}=1 found")
    if len(samples_0) == 0:
        raise ValueError(f"No samples with {label}=0 found")
    
    print(f"Label=1 samples: {len(samples_1)}, Label=0 samples: {len(samples_0)}")
    
    X_dummy = np.arange(len(samples_1))
    rkf = RepeatedKFold(n_splits=n_folds, n_repeats=n_repeats, random_state=random_state)
    
    dataset_samples_list = []
    
    for fold_idx, (train_idx, test_idx) in enumerate(rkf.split(X_dummy)):
        fold_samples_1 = [samples_1[i] for i in test_idx]
        dataset_samples = fold_samples_1 + samples_0
        dataset_samples_list.append(dataset_samples)
        repeat_idx = fold_idx // n_folds
        fold_in_repeat = fold_idx % n_folds
        print(f"Repeat {repeat_idx + 1}, Fold {fold_in_repeat + 1}: {len(fold_samples_1)} samples from class 1 + {len(samples_0)} samples from class 0 = {len(dataset_samples)} total")
    
    print(f"Total datasets created: {len(dataset_samples_list)}")
    return dataset_samples_list


def preprocess_data(
    df: pd.DataFrame,
    metadata: pd.DataFrame,
    datatype: str,
    output_dir: str,
    label: str,
    n_cpus: int = 8,
):
    """
    根据数据类型对数据进行预处理
    
    Args:
        df: pd.DataFrame, (n_samples, n_features)
        metadata: pd.DataFrame, 包含label列的元数据
        datatype: str, 数据类型
        output_dir: str, 输出目录
        label: str, label列名
        n_cpus: int, 使用的CPU核心数 (default: 1)
    
    Returns:
        pd.DataFrame: 预处理后的数据，用于后续差异分析
    """
    if datatype not in ["artemis", "end_motif"]:
        df = filter_features(df, min_sum=10.0)
        normalized_counts, vst_counts = transform_counts(df, metadata, label, n_cpus=n_cpus)
        normalized_counts.to_csv(os.path.join(output_dir, "normalized_counts.tsv"), sep="\t", index=True)
        vst_counts.to_csv(os.path.join(output_dir, "vst_counts.tsv"), sep="\t", index=True)
        analysis_df = df

    elif datatype == "artemis":
        row_zero_ratio = (df == 0).sum(axis=1) / df.shape[1]
        col_zero_ratio = (df == 0).sum(axis=0) / df.shape[0]
        zero_threshold = 0.8
        rows_to_keep = row_zero_ratio <= zero_threshold
        cols_to_keep = col_zero_ratio <= zero_threshold
        df = df.loc[rows_to_keep, cols_to_keep]
        df_log1p = np.log1p(df)
        df_log1p.to_csv(os.path.join(output_dir, "artemis_log1p.tsv"), sep="\t", index=True)
        df.to_csv(os.path.join(output_dir, "artemis.tsv"), sep="\t", index=True)
        analysis_df = df_log1p
    
    elif datatype == "end_motif":
        df.to_csv(os.path.join(output_dir, "end_motif_matrix.tsv"), sep="\t", index=True)
        analysis_df = df
    
    return analysis_df


def differential_analysis(
    analysis_df: pd.DataFrame,
    metadata: pd.DataFrame,
    dataset_samples_list: list,
    datatype: str,
    output_dir: str,
    label: str,
    threshold: float = None,
    qvalue_threshold: float = 0.05,
    n_cpus: int = 8,
):
    """
    对9个dataset分别进行差异分析，统计每个特征在不同dataset中的出现情况
    
    Args:
        analysis_df: pd.DataFrame, (n_samples, n_features) 预处理后的数据
        metadata: pd.DataFrame, 包含label列的元数据
        dataset_samples_list: list, 包含9个dataset的样本列表
        datatype: str, 数据类型
        output_dir: str, 输出目录
        label: str, label列名
        threshold: float, 阈值（用于counts/tpm时作为logfc_threshold，用于prop时作为coefficient_threshold）
        qvalue_threshold: float, qvalue阈值（default: 0.05）
        n_cpus: int, 使用的CPU核心数 (default: 1)
    
    Returns:
        pd.DataFrame: 二进制矩阵，行是特征，列是dataset（包含n_datasets统计列）
    """
    
    all_sig_features_list = []
    
    if threshold is None:
        threshold = 0.0
        threshold_name = "logfc_threshold" if datatype != "end_motif" else "coefficient_threshold"
        print(f"Warning: threshold not provided, using default value 0.0 as {threshold_name}")
    
    for dataset_idx, dataset_samples in enumerate(dataset_samples_list):
        print(f"\n{'='*60}")
        print(f"Processing dataset {dataset_idx + 1}/9")
        print(f"{'='*60}")
        
        dataset_df = analysis_df.loc[dataset_samples, :]
        dataset_metadata = metadata.loc[dataset_samples, :]
        
        dataset_output_dir = os.path.join(output_dir, f"dataset_{dataset_idx + 1}")
        os.makedirs(dataset_output_dir, exist_ok=True)
        
        if datatype == "artemis":
            # artemis使用differential_analysis_tpm
            sig_features = differential_analysis_tpm(
                df=dataset_df,
                metadata=dataset_metadata,
                output_dir=dataset_output_dir,
                logfc_threshold=threshold,
                label=label,
                qvalue_threshold=qvalue_threshold,
            )
        elif datatype == "end_motif":
            # end_motif使用differential_analysis_prop，使用threshold作为coefficient_threshold
            sig_features = differential_analysis_prop(
                df=dataset_df,
                metadata=dataset_metadata,
                output_dir=dataset_output_dir,
                label=label,
                coefficient_threshold=threshold,
                qvalue_threshold=qvalue_threshold,
            )
        else:
            # 其他类型使用differential_analysis_counts
            sig_features = differential_analysis_counts(
                df=dataset_df,
                metadata=dataset_metadata,
                output_dir=dataset_output_dir,
                logfc_threshold=threshold,
                label=label,
                qvalue_threshold=qvalue_threshold,
                n_cpus=n_cpus,
            )
        
        all_sig_features_list.append(set(sig_features))
        print(f"Dataset {dataset_idx + 1}: Found {len(sig_features)} significant features")
    
    all_features = set()
    for sig_features_set in all_sig_features_list:
        all_features.update(sig_features_set)
    
    all_features_list = sorted(list(all_features))
    
    binary_matrix = pd.DataFrame(
        0,
        index=all_features_list,
        columns=[f"dataset_{i+1}" for i in range(len(dataset_samples_list))]
    )
    
    for dataset_idx, sig_features_set in enumerate(all_sig_features_list):
        if len(sig_features_set) > 0:
            binary_matrix.loc[list(sig_features_set), f"dataset_{dataset_idx + 1}"] = 1
    
    binary_matrix["n_datasets"] = binary_matrix.iloc[:, :len(dataset_samples_list)].sum(axis=1)
    binary_matrix = binary_matrix.sort_values("n_datasets", ascending=False)
    
    binary_matrix_path = os.path.join(output_dir, "sig_features_binary_matrix.tsv")
    binary_matrix.to_csv(binary_matrix_path, sep="\t", index=True)
    return binary_matrix


def get_intersection_features(
    binary_matrix: pd.DataFrame,
    n_datasets: int = None,
):
    """
    从二进制矩阵中提取在所有dataset中都出现的特征
    
    Args:
        binary_matrix: pd.DataFrame, 二进制矩阵，包含n_datasets列
        n_datasets: int, 需要的dataset数量（默认None，表示所有dataset，即9个）
    
    Returns:
        list: 在所有dataset中都出现的特征列表
    """
    if n_datasets is None:
        dataset_cols = [col for col in binary_matrix.columns if col.startswith('dataset_')]
        n_datasets = len(dataset_cols)
    
    intersection_features = binary_matrix[binary_matrix['n_datasets'] >= n_datasets].index.tolist()
    
    return intersection_features


def extract_effect_matrix(
    intersection_features: list,
    output_dir: str,
    datatype: str,
    n_datasets: int = 9,
    output_filename: str = "features_effect_matrix.tsv",
):
    """
    从每个dataset的差异分析结果中提取intersection_features对应的effect值（logfc或coefficient）
    创建一个intersection_features * n_datasets的矩阵
    
    Args:
        intersection_features: list, 在所有dataset中都出现的特征列表
        output_dir: str, 输出目录（包含dataset_1到dataset_n的子目录）
        datatype: str, 数据类型（决定使用log2FoldChange还是coefficient）
        n_datasets: int, dataset数量（default: 9）
        output_filename: str, 输出文件名（default: "intersection_features_effect_matrix.tsv"）
    
    Returns:
        pd.DataFrame: effect矩阵，行是intersection_features，列是dataset
    """
    if len(intersection_features) == 0:
        print("Warning: No intersection features found, returning empty matrix")
        return pd.DataFrame()
    
    # 根据datatype确定effect列名
    if datatype == "end_motif":
        effect_col = "coefficient"
    else:
        effect_col = "log2FoldChange"
    
    effect_matrix = pd.DataFrame(
        index=intersection_features,
        columns=[f"dataset_{i+1}" for i in range(n_datasets)]
    )
    
    for dataset_idx in range(n_datasets):
        dataset_dir = os.path.join(output_dir, f"dataset_{dataset_idx + 1}")
        results_file = os.path.join(dataset_dir, "diff_results.tsv")
        
        if not os.path.exists(results_file):
            raise FileNotFoundError(f"Results file not found: {results_file}")
        
        if datatype == "end_motif":
            results_df = pd.read_csv(results_file, sep="\t", index_col="feature")
        else:
            results_df = pd.read_csv(results_file, sep="\t", index_col=0)
        
        effect_matrix.loc[intersection_features, f"dataset_{dataset_idx + 1}"] = results_df.loc[intersection_features, effect_col].values
        
    effect_matrix = effect_matrix.astype(float)
    
    output_path = os.path.join(output_dir, output_filename)
    effect_matrix.to_csv(output_path, sep="\t", index=True)
    print(f"Saved effect matrix to {output_path}")
    print(f"Effect matrix shape: {effect_matrix.shape}")
    
    return effect_matrix


def extract_log10qvalue_matrix(
    intersection_features: list,
    output_dir: str,
    datatype: str,
    n_datasets: int = 9,
    output_filename: str = "features_log10qvalue_matrix.tsv",
):
    """
    从每个dataset的差异分析结果中提取intersection_features对应的-log10qvalue值
    创建一个intersection_features * n_datasets的矩阵
    
    Args:
        intersection_features: list, 在所有dataset中都出现的特征列表
        output_dir: str, 输出目录（包含dataset_1到dataset_n的子目录）
        datatype: str, 数据类型
        n_datasets: int, dataset数量（default: 9）
        output_filename: str, 输出文件名（default: "intersection_features_log10qvalue_matrix.tsv"）
    
    Returns:
        pd.DataFrame: -log10qvalue矩阵，行是intersection_features，列是dataset
    """
    if len(intersection_features) == 0:
        print("Warning: No intersection features found, returning empty matrix")
        return pd.DataFrame()
    
    log10qvalue_matrix = pd.DataFrame(
        index=intersection_features,
        columns=[f"dataset_{i+1}" for i in range(n_datasets)]
    )
    
    for dataset_idx in range(n_datasets):
        dataset_dir = os.path.join(output_dir, f"dataset_{dataset_idx + 1}")
        results_file = os.path.join(dataset_dir, "diff_results.tsv")
        
        if not os.path.exists(results_file):
            raise FileNotFoundError(f"Results file not found: {results_file}")
        
        if datatype == "end_motif":
            results_df = pd.read_csv(results_file, sep="\t", index_col="feature")
        else:
            results_df = pd.read_csv(results_file, sep="\t", index_col=0)
        
        log10qvalue_matrix.loc[intersection_features, f"dataset_{dataset_idx + 1}"] = results_df.loc[intersection_features, "-log10qvalue"].values
    
    log10qvalue_matrix = log10qvalue_matrix.astype(float)
    
    output_path = os.path.join(output_dir, output_filename)
    log10qvalue_matrix.to_csv(output_path, sep="\t", index=True)
    print(f"Saved -log10qvalue matrix to {output_path}")
    print(f"-log10qvalue matrix shape: {log10qvalue_matrix.shape}")
    
    return log10qvalue_matrix


def combine_matrices(
    effect_matrix: pd.DataFrame,
    log10qvalue_matrix: pd.DataFrame,
    output_dir: str,
    output_filename: str = "feature_analysis_matrix.tsv",
):
    """
    合并effect_matrix和log10qvalue_matrix，计算统计量
    
    Args:
        effect_matrix: pd.DataFrame, effect矩阵（logfc或coefficient）
        log10qvalue_matrix: pd.DataFrame, -log10qvalue矩阵
        output_dir: str, 输出目录
        output_filename: str, 输出文件名（default: "feature_summary_matrix.tsv"）
    
    Returns:
        pd.DataFrame: 综合矩阵，包含effect_mean, effect_CV, mean_log10qvalue三列
    """
    effect_signs = np.sign(effect_matrix)
    all_non_negative = (effect_signs >= 0).all(axis=1)
    all_non_positive = (effect_signs <= 0).all(axis=1)
    consistent_sign_mask = all_non_negative | all_non_positive
    
    effect_matrix_filtered = effect_matrix.loc[consistent_sign_mask]
    log10qvalue_matrix_filtered = log10qvalue_matrix.loc[consistent_sign_mask]
    
    print(f"Filtered out {len(effect_matrix) - len(effect_matrix_filtered)} features with inconsistent effect signs")
    print(f"Remaining features: {len(effect_matrix_filtered)}")
    
    effect_mean = effect_matrix_filtered.mean(axis=1)
    effect_std = effect_matrix_filtered.std(axis=1)
    effect_cv = effect_std / effect_mean.abs()  # 变异系数 = 标准差 / |均值|
    log10qvalue_mean = log10qvalue_matrix_filtered.mean(axis=1)
    
    summary_matrix = pd.DataFrame({
        'effect_mean': effect_mean,
        'effect_CV': effect_cv,
        'mean_log10qvalue': log10qvalue_mean
    })
    
    summary_matrix = summary_matrix.sort_values("effect_mean", ascending=False)
    output_path = os.path.join(output_dir, output_filename)
    summary_matrix.to_csv(output_path, sep="\t", index=True)
    print(f"Saved feature summary matrix to {output_path}")
    
    return summary_matrix


def plot_feature_scatter(
    summary_matrix: pd.DataFrame,
    output_dir: str,
    output_filename: str = "feature_analysis_scatter.png",
):
    """
    绘制散点图：横坐标为effect_size的均值，纵坐标为effect_size的变异系数，
    颜色为log10qvalue的均值（值越小颜色越浅）
    
    Args:
        summary_matrix: pd.DataFrame, 综合矩阵，包含effect_mean, effect_CV, mean_log10qvalue列
        output_dir: str, 输出目录
        output_filename: str, 输出文件名（default: "feature_analysis_scatter.png"）
    """
    
    fig, ax = plt.subplots(figsize=(10, 8))
    scatter = ax.scatter(
        summary_matrix['effect_mean'],
        summary_matrix['effect_CV'],
        c=summary_matrix['mean_log10qvalue'],
        cmap='viridis',  # 使用viridis配色，值越小颜色越浅（从黄色到深蓝）
        alpha=0.6,
        s=50,
        edgecolors='black',
        linewidths=0.5,
    )
    
    # 添加颜色条
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Mean -log10(qvalue)', fontsize=12, fontweight='bold')
    
    # 设置标签和标题
    ax.set_xlabel('Mean Effect Size', fontsize=12, fontweight='bold')
    ax.set_ylabel('Coefficient of Variation (CV)', fontsize=12, fontweight='bold')
    ax.set_title('Effect Size vs Variability (colored by -log10qvalue)', fontsize=14, fontweight='bold')
    
    # 在x=0处添加垂直线
    ax.axvline(x=0, color='red', linestyle='--', linewidth=1.5, alpha=0.7, label='x=0')
    
    # 添加网格
    ax.grid(True, alpha=0.3)
    ax.set_axisbelow(True)
    
    # 调整布局
    plt.tight_layout()
    
    # 保存图片
    scatter_path = os.path.join(output_dir, output_filename)
    plt.savefig(scatter_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved effect vs variability scatter plot to {scatter_path}")
    
    return None


def plot_feature_count_barplot(
    binary_matrix: pd.DataFrame,
    output_dir: str,
    output_filename: str = "feature_count_barplot.png",
):
    """
    绘制条形图，统计不同出现次数下特征个数
    
    Args:
        binary_matrix: pd.DataFrame, 二进制矩阵，包含n_datasets列
        output_dir: str, 输出目录
        output_filename: str, 输出文件名（default: "feature_count_barplot.png"）
    """
    count_stats = binary_matrix['n_datasets'].value_counts().sort_index()
    
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(count_stats.index, count_stats.values, color='steelblue', alpha=0.7, edgecolor='black')
    
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax.set_xlabel('Number of Datasets (n_datasets)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Number of Features', fontsize=12, fontweight='bold')
    ax.set_title('Distribution of Features Across Datasets', fontsize=14, fontweight='bold')
    
    ax.set_xticks(count_stats.index)
    ax.set_xticklabels(count_stats.index, fontsize=10)
    
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_axisbelow(True)
    
    plt.tight_layout()
    
    barplot_path = os.path.join(output_dir, output_filename)
    plt.savefig(barplot_path, dpi=300, bbox_inches='tight')
    plt.close()
    return None


# def plot_binary_matrix_heatmap(
#     binary_matrix: pd.DataFrame,
#     output_dir: str,
#     output_filename: str = "sig_features_heatmap.png",
# ):
#     """
#     对二进制矩阵绘制热图，行聚类，列不聚类
    
#     Args:
#         binary_matrix: pd.DataFrame, 二进制矩阵，包含dataset列和n_datasets列
#         output_dir: str, 输出目录
#         output_filename: str, 输出文件名（default: "sig_features_heatmap.png"）
#     """
#     dataset_cols = [col for col in binary_matrix.columns if col.startswith('dataset_')]
#     heatmap_data = binary_matrix[dataset_cols]
    
#     fig_height = 12
    
#     g = sns.clustermap(
#         heatmap_data,
#         row_cluster=True,
#         col_cluster=False,
#         cmap='RdYlBu_r',  # 红-黄-蓝配色，反转后0为蓝色，1为红色
#         vmin=0,
#         vmax=1,
#         figsize=(8, fig_height),
#         cbar_kws={'label': 'Significant (1) / Not Significant (0)'},
#         xticklabels=False,
#         yticklabels=False,
#     )
    
#     g.fig.suptitle('Significant Features Across Datasets', y=1.02, fontsize=14, fontweight='bold')
#     heatmap_path = os.path.join(output_dir, output_filename)
#     g.savefig(heatmap_path, dpi=300, bbox_inches='tight')
#     plt.close()
#     return None


def filter_features(df: pd.DataFrame, min_sum: float = 10.0) -> pd.DataFrame:
    """
    Filter features (columns) by sum count threshold

    Args:
        df: pd.DataFrame (samples, features)
        min_sum: Minimum sum threshold for a feature to be kept (default: 5.0)

    Returns:
        pd.DataFrame: Filtered dataframe with only features that have sum > min_sum
    """
    n_features_before = len(df.columns)
    col_sums = df.sum(axis=0)
    valid_cols = col_sums[col_sums > min_sum].index
    df_filtered = df.loc[:, valid_cols]
    n_filtered = n_features_before - len(valid_cols)
    if n_filtered > 0:
        print(f"Filtered out {n_filtered} features with sum_count <= {min_sum}")
    print(f"Remaining features: {len(df_filtered.columns)}")
    return df_filtered


def main(args):    
    if args.datatype == "gene_counts":
        df = pd.read_csv(args.input, sep="\t", index_col=0, header=0)
        df = df.T
        
    elif args.datatype == "artemis":
        df = pd.read_csv(args.input, index_col=0, header=0)
        df.rename(columns={'id': 'sample'}, inplace=True)
        df.set_index('sample', inplace=True)
        
    elif args.datatype == "end_motif":
        df = pd.read_csv(args.input, sep="\t", header=0, index_col=0)
        df = df.T
        
    elif args.datatype == "FSD":
        df = pd.read_csv(args.input, sep="\t", header=0, index_col=0)

    elif args.datatype in ["consensus_peak", "OCR", "window"]:
        df = pd.read_csv(args.input, sep="\t")
        if "chrom" in df.columns:
            df.rename(columns={"chrom": "chr"}, inplace=True)
        
        df["region"] = df["chr"].astype(str) + "_" + df["start"].astype(str) + "_" + df["end"].astype(str)
        df.set_index("region", inplace=True)
        df.drop(columns=["chr", "start", "end"], inplace=True)
        df = df.T
    else:
        raise ValueError(f"Invalid datatype: {args.datatype}")
    
    # now every df is (sample, feature)       
    metadata = pd.read_csv(args.metadata)
    if "sample" not in metadata.columns:
        raise ValueError("metadata must contain 'sample' column")
    if args.label not in metadata.columns:
        raise ValueError(f"metadata must contain '{args.label}' column")

    metadata = metadata[metadata[args.label].notna()]
    valid_samples = metadata["sample"].unique()
    metadata.set_index("sample", inplace=True)
    metadata[args.label] = metadata[args.label].astype(int).astype(str)
    metadata["batch"] = metadata["batch"].astype(str)
    metadata["sex"] = metadata["sex"].astype(int)
    metadata["age"] = metadata["age"].astype(int)
    dataset_samples_list = split_exp_set(metadata, args.label, n_folds=3, n_repeats=3, random_state=42)
    
    df = df.loc[valid_samples, :]
    
    n_threads = getattr(args, 'threads', 8)
    
    analysis_df = preprocess_data(df, metadata, args.datatype, args.output_dir, args.label, n_cpus=n_threads)
    
    binary_matrix = differential_analysis(
        analysis_df=analysis_df,
        metadata=metadata,
        dataset_samples_list=dataset_samples_list,
        datatype=args.datatype,
        output_dir=args.output_dir,
        label=args.label,
        threshold=args.logfc_threshold,
        qvalue_threshold=args.qvalue_threshold,
        n_cpus=n_threads,
    )
    
    print(f"\nBinary matrix shape: {binary_matrix.shape}")
    print(f"Features appearing in all 9 datasets: {len(binary_matrix[binary_matrix['n_datasets'] == 9])}")
    
    plot_feature_count_barplot(binary_matrix, args.output_dir)
    
    # plot_binary_matrix_heatmap(binary_matrix, args.output_dir)
    
    intersection_features = get_intersection_features(binary_matrix)
    if len(intersection_features) > 0:
        effect_matrix = extract_effect_matrix(
            intersection_features=intersection_features,
            output_dir=args.output_dir,
            datatype=args.datatype,
            n_datasets=len(dataset_samples_list),
        )
        
        log10qvalue_matrix = extract_log10qvalue_matrix(
            intersection_features=intersection_features,
            output_dir=args.output_dir,
            datatype=args.datatype,
            n_datasets=len(dataset_samples_list),
        )
        
        summary_matrix = combine_matrices(
            effect_matrix=effect_matrix,
            log10qvalue_matrix=log10qvalue_matrix,
            output_dir=args.output_dir,
        )
        
        plot_feature_scatter(
            summary_matrix=summary_matrix,
            output_dir=args.output_dir
        )
    else:
        print("No intersection features found")  
          
    return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", required=True, help="path to input file", type=str)
    parser.add_argument("-t", "--datatype", required=True, help="datatype", type=str,
                        choices=["artemis", "end_motif", "FSD", "gene_counts", "consensus_peak", "OCR", "window"])
    parser.add_argument("-m", "--metadata", required=True, help="path to metadata file", type=str)
    parser.add_argument("-o", "--output_dir", default="./", help="path to output directory", type=str)
    parser.add_argument("-l", "--label", required=True, help="label", type=str)
    parser.add_argument("-fc", "--logfc_threshold", type=float, default=None, help="logfc threshold or coefficient threshold")
    parser.add_argument("-q", "--qvalue_threshold", type=float, default=0.05, help="qvalue threshold (default: 0.05)")
    parser.add_argument("--threads", type=int, default=8, help="number of CPU threads to use (default: 8)")
    args = parser.parse_args()
    main(args)
