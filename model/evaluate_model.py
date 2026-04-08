import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import joblib
from sklearn.metrics import auc, precision_recall_curve, roc_auc_score, roc_curve
import argparse


def grouped_permutation_importance(
    model,
    X_test,
    y_test,
    idxs,
    group_names,
    n_repeats=30,
    random_state=42,
    scoring="roc_auc"
):
    """
    计算分组排列重要性

    参数:
        model: 训练好的模型
        X_test: 测试集特征
        y_test: 测试集标签
        idxs: 特征分组的索引列表，例如 [[0,1], [2,3], [4,5]]
        group_names: 每个分组的名称列表，例如 ['age', 'bmi', 'sex']
        n_repeats: 重复次数
        random_state: 随机种子
        scoring: 评分方法，支持 "roc_auc" 或 "pr_auc"

    返回:
        dict: 键为 group_name，值为包含 importance, baseline 的字典
    """
    np.random.seed(random_state)

    y_prob = model.predict_proba(X_test)[:, 1]
    if scoring == "roc_auc":
        baseline_score = roc_auc_score(y_test, y_prob)
    elif scoring == "pr_auc":
        precision, recall, _ = precision_recall_curve(y_test, y_prob)
        baseline_score = auc(recall, precision)
    else:
        raise ValueError(f"Unsupported scoring: {scoring}")

    results = {}
    for group_name, group_idx in zip(group_names, idxs):
        group_scores = []
        for _ in range(n_repeats):
            X_permuted = X_test.copy()
            for idx in group_idx:
                permutation = np.random.permutation(len(X_test))
                X_permuted[:, idx] = X_test[permutation, idx]

            y_prob_permuted = model.predict_proba(X_permuted)[:, 1]
            if scoring == "roc_auc":
                permuted_score = roc_auc_score(y_test, y_prob_permuted)
            elif scoring == "pr_auc":
                precision, recall, _ = precision_recall_curve(
                    y_test, y_prob_permuted)
                permuted_score = auc(recall, precision)

            # importance = baseline - permuted
            group_scores.append(baseline_score - permuted_score)

        results[group_name] = {
            'importance': group_scores,
            'baseline': baseline_score
        }

    return results


def parse_feature_list(feature_list):
    """
    解析特征列表，提取特征类型和类别信息

    参数:
        feature_list: 特征名称列表，格式为 {feature_type}_proba_0 / _proba_1 / _proba_2

    返回:
        df: DataFrame，包含 index, feature_name, feature_type, class 四列
    """
    feature_info = []
    for idx, feat_name in enumerate(feature_list):
        if feat_name.endswith('_proba_0'):
            feat_type = feat_name.replace('_proba_0', '')
            class_label = '0'
        elif feat_name.endswith('_proba_1'):
            feat_type = feat_name.replace('_proba_1', '')
            class_label = '1'
        elif feat_name.endswith('_proba_2'):
            feat_type = feat_name.replace('_proba_2', '')
            class_label = '2'
        else:
            raise ValueError(f"{feat_name} is not a valid feature name")

        feature_info.append({
            'index': idx,
            'feature_name': feat_name,
            'feature_type': feat_type,
            'class': class_label
        })

    return pd.DataFrame(feature_info)


def feature_importance_boxplot(
    model_list,
    feature_list,
    output_path
):
    """
    绘制模型特征系数的箱线图

    参数:
        model_list: 包含多个模型的列表，每个模型都有coef_属性
        feature_list: 特征名称列表，包含三类特征，每类有0类和1类预测概率
        output_path: 输出图片路径
    """
    coef_matrix = []
    for model in model_list:
        coef_matrix.append(np.abs(model.coef_[0]))
    coef_matrix = np.array(coef_matrix)  # shape: (n_models, n_features)

    feature_info_df = parse_feature_list(feature_list)

    n_models = len(model_list)
    df = pd.concat([feature_info_df] * n_models, ignore_index=True)
    df['coef'] = coef_matrix.T.ravel()

    feature_types = df['feature_type'].unique()

    # 使用matplotlib内置色板
    cmap = plt.get_cmap('tab10')
    colors = [cmap(i) for i in range(len(feature_types))]

    plt.figure(figsize=(6, 6))
    positions = []
    data_to_plot = []
    labels = []

    for i, feat_type in enumerate(feature_types):
        mask = df['feature_type'] == feat_type
        combined_data = df[mask]['coef'].values

        positions.append(i)
        data_to_plot.append(combined_data)
        labels.append(feat_type)

    bp = plt.boxplot(data_to_plot, positions=positions, widths=0.6,
                     patch_artist=True, showfliers=True,
                     boxprops=dict(linewidth=1.5),
                     medianprops=dict(color='black', linewidth=2),
                     whiskerprops=dict(linewidth=1.5),
                     capprops=dict(linewidth=1.5))

    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    # 在箱线图上叠加散点
    for i, (pos, data) in enumerate(zip(positions, data_to_plot)):
        # 添加水平抖动(jitter)使散点不完全重叠
        jitter = np.random.normal(0, 0.04, size=len(data))
        x_data = np.ones(len(data)) * pos + jitter
        plt.scatter(x_data, data, alpha=0.4, s=20, color=colors[i], edgecolors='none')

    plt.xticks(positions, labels, fontsize=11, rotation=45, ha='right')
    plt.ylabel('Coefficient (|coef|)', fontsize=12, fontweight='bold')
    plt.xlabel('Feature Type', fontsize=12, fontweight='bold')
    plt.title('Feature Coefficient Distribution', fontsize=14,
              fontweight='bold', pad=20)

    # 添加网格线
    plt.grid(axis='y', alpha=0.3, linestyle='--')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    return None


def load_cv_matrix(feature_dir: str) -> pd.DataFrame:
    path = os.path.join(feature_dir, "aggregate_df.tsv")
    df = pd.read_csv(path, sep="\t", header=0)
    return df


def get_model_list(
    feature_dir: str,
    cv_matrix: pd.DataFrame,
    n_folds: int = 9,
):
    model_list = []
    X_test_list = []
    y_test_list = []
    test_index_list = []
    ml_results_dir = os.path.join(feature_dir, "ML_results")
    for fold_idx in range(n_folds):
        fold_dir = os.path.join(ml_results_dir, f"fold_{fold_idx}")
        model_path = os.path.join(fold_dir, "best_model.pkl")
        model = joblib.load(model_path)
        model_list.append(model)
        X_test_path = os.path.join(fold_dir, "X_test.npy")
        y_test_path = os.path.join(fold_dir, "y_test.npy")
        X_test = np.load(X_test_path)
        y_test = np.load(y_test_path)
        X_test_list.append(X_test)
        y_test_list.append(y_test)
        test_index_path = os.path.join(fold_dir, "test_index.npy")
        test_index = np.load(test_index_path)
        test_index_list.append(test_index)

    cv_matrix_dropped = cv_matrix.drop(columns=["test_index", "y_test", "stage"])
    fallback_feature_list = cv_matrix_dropped.columns.tolist()
    feature_names_path = os.path.join(ml_results_dir, "fold_0", "feature_names.npy")
    if os.path.isfile(feature_names_path):
        feature_list = np.load(feature_names_path, allow_pickle=True).tolist()
    else:
        n_feat = X_test_list[0].shape[1]
        feature_list = fallback_feature_list[:n_feat] if len(fallback_feature_list) >= n_feat else fallback_feature_list
    return model_list, X_test_list, y_test_list, test_index_list, feature_list


def add_stage_to_result(result: pd.DataFrame, cv_matrix: pd.DataFrame) -> pd.DataFrame:
    stage_df = (
        cv_matrix[["test_index", "y_test", "stage"]]
        .drop_duplicates("test_index")
        .sort_values("test_index")
        .reset_index(drop=True)
    )
    if len(result) != len(stage_df):
        raise ValueError(
            f"Length mismatch: result has {len(result)} rows, "
            f"stage source has {len(stage_df)} rows"
        )
    if (result["y_test"].values != stage_df["y_test"].values).any():
        bad = result["y_test"] != stage_df["y_test"]
        raise ValueError(
            "result['y_test'] and cv_matrix (stage source) 'y_test' are not identical. "
            f"Mismatch at rows: {bad[bad].index.tolist()}"
        )
    result = result.copy()
    result["stage"] = stage_df["stage"].values
    return result


def get_fold_proba_matrix(
    model_list: list,
    X_test_list: list,
    y_test_list: list,
    test_index_list: list,
    cv_matrix: pd.DataFrame,
    n_repeats: int = 3,
) -> pd.DataFrame:
    n_folds = len(model_list)
    if n_folds % n_repeats != 0:
        raise ValueError(
            f"n_folds ({n_folds}) must be divisible by n_repeats ({n_repeats})"
        )
    folds_per_repeat = n_folds // n_repeats

    result = None
    for r in range(n_repeats):
        start_fold = r * folds_per_repeat
        end_fold = (r + 1) * folds_per_repeat
        test_index_r = np.concatenate(
            [test_index_list[i] for i in range(start_fold, end_fold)]
        )
        y_test_r = np.concatenate(
            [y_test_list[i] for i in range(start_fold, end_fold)]
        )
        prob_r = np.concatenate([
            model_list[i].predict_proba(X_test_list[i])[:, 1]
            for i in range(start_fold, end_fold)
        ])
        sort_idx = np.argsort(test_index_r)
        test_index_r = test_index_r[sort_idx]
        prob_r = prob_r[sort_idx]
        if result is None:
            y_test_r = y_test_r[sort_idx]
            result = pd.DataFrame({
                "test_index": test_index_r,
                "y_test": y_test_r,
                f"prob_1_repeat_{r}": prob_r,
            })
        else:
            result[f"prob_1_repeat_{r}"] = prob_r
    result = result.reset_index(drop=True)
    result = add_stage_to_result(result, cv_matrix)
    return result


def get_auc_plot(
    model_list,
    X_test_list,
    y_test_list,
    output_path
):
    """
    绘制ROC曲线的平均值和误差带

    参数:
        model_list: 模型列表
        X_test_list: 测试集特征列表
        y_test_list: 测试集标签列表
        output_path: 输出图片路径
    """
    tprs = []
    aucs_roc = []
    mean_fpr = np.linspace(0, 1, 100)

    for model, X_test, y_test in zip(model_list, X_test_list, y_test_list):
        y_prob = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        tprs.append(np.interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0
        roc_auc = roc_auc_score(y_test, y_prob)
        aucs_roc.append(roc_auc)

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    std_tpr = np.std(tprs, axis=0)
    mean_auc_roc = np.mean(aucs_roc)
    std_auc_roc = np.std(aucs_roc)

    fig, ax1 = plt.subplots(1, 1, figsize=(6, 6))
    ax1.plot(mean_fpr, mean_tpr, color='#FF6B6B', linewidth=2.5,
             label=f'Mean ROC (AUC = {mean_auc_roc:.3f} ± {std_auc_roc:.3f})')
    ax1.fill_between(mean_fpr, mean_tpr - std_tpr, mean_tpr + std_tpr,
                     color='#FF6B6B', alpha=0.2, label='± 1 std')
    ax1.plot([0, 1], [0, 1], 'k--', linewidth=1.5, alpha=0.5,
             label='random guess')

    ax1.set_xlabel('False Positive Rate', fontsize=12, fontweight='bold')
    ax1.set_ylabel('True Positive Rate', fontsize=12, fontweight='bold')
    ax1.set_title('ROC Curve', fontsize=14, fontweight='bold', pad=15)
    ax1.legend(loc='lower right', fontsize=10, framealpha=0.9)
    ax1.grid(alpha=0.3, linestyle='--')
    ax1.set_xlim([-0.05, 1.05])
    ax1.set_ylim([-0.05, 1.05])

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    output_dir = os.path.dirname(output_path)
    plot_data = {
        'roc': {
            'mean_fpr': mean_fpr,
            'mean_tpr': mean_tpr,
            'std_tpr': std_tpr,
            'mean_auc': mean_auc_roc,
            'std_auc': std_auc_roc
        }
    }
    data_path = os.path.join(output_dir, 'auc_plot_data.pkl')
    joblib.dump(plot_data, data_path, compress=3, protocol=4)
    return None


def reload_and_plot_auc(data_path, output_path):
    """
    从保存的数据重新绘制ROC曲线

    参数:
        data_path: auc_plot_data.pkl 文件路径
        output_path: 输出图片路径
    """
    plot_data = joblib.load(data_path)
    mean_fpr = plot_data['roc']['mean_fpr']
    mean_tpr = plot_data['roc']['mean_tpr']
    std_tpr = plot_data['roc']['std_tpr']
    mean_auc_roc = plot_data['roc']['mean_auc']
    std_auc_roc = plot_data['roc']['std_auc']

    fig, ax1 = plt.subplots(1, 1, figsize=(6, 6))
    ax1.plot(mean_fpr, mean_tpr, color='#FF6B6B', linewidth=2.5,
             label=f'Mean ROC (AUC = {mean_auc_roc:.3f} ± {std_auc_roc:.3f})')
    ax1.fill_between(mean_fpr, mean_tpr - std_tpr, mean_tpr + std_tpr,
                     color='#FF6B6B', alpha=0.2, label='± 1 std')
    ax1.plot([0, 1], [0, 1], 'k--', linewidth=1.5, alpha=0.5,
             label='random guess')

    ax1.set_xlabel('False Positive Rate', fontsize=12, fontweight='bold')
    ax1.set_ylabel('True Positive Rate', fontsize=12, fontweight='bold')
    ax1.set_title('ROC Curve', fontsize=14, fontweight='bold', pad=15)
    ax1.legend(loc='lower right', fontsize=10, framealpha=0.9)
    ax1.grid(alpha=0.3, linestyle='--')
    ax1.set_xlim([-0.05, 1.05])
    ax1.set_ylim([-0.05, 1.05])

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    return None


def permutation_importance(
    model_list,
    feature_list,
    X_test_list,
    y_test_list,
    scoring="roc_auc",
    n_repeats=30
):
    feature_info = parse_feature_list(feature_list)
    grouped = feature_info.groupby('feature_type', sort=False)
    group_idxs = [group['index'].tolist() for _, group in grouped]
    group_names = [name for name, _ in grouped]

    results_list = []
    for model, X_test, y_test in zip(model_list, X_test_list, y_test_list):
        results = grouped_permutation_importance(
            model,
            X_test,
            y_test,
            idxs=group_idxs,
            group_names=group_names,
            n_repeats=n_repeats,
            random_state=42,
            scoring=scoring
        )
        results_list.append(results)

    merged_results = {}
    for group_name in group_names:
        all_importances = []
        all_baselines = []

        for fold_result in results_list:
            all_importances.extend(fold_result[group_name]['importance'])
            all_baselines.append(fold_result[group_name]['baseline'])

        merged_results[group_name] = {
            'importance': all_importances,
            'mean': np.mean(all_importances),
            'std': np.std(all_importances),
            'baseline': all_baselines
        }

    return merged_results


def plot_permutation_importance(
    perm_results,
    output_path,
):
    """
    绘制排列重要性柱状图

    参数:
        perm_results: permutation_importance 函数返回的结果字典
        output_path: 输出图片路径
    """
    group_names = list(perm_results.keys())
    means = [perm_results[name]['mean'] for name in group_names]
    stds = [perm_results[name]['std'] for name in group_names]

    # 使用matplotlib内置色板
    cmap = plt.get_cmap('tab10')
    colors = [cmap(i) for i in range(len(group_names))]

    # 创建图形
    fig, ax = plt.subplots(figsize=(6, 6))

    # 绘制柱状图
    x_pos = np.arange(len(group_names))
    bars = ax.bar(x_pos, means, yerr=stds,
                  color=colors,
                  alpha=0.8, capsize=8, width=0.6,
                  error_kw={'linewidth': 2, 'ecolor': 'black', 'alpha': 0.7})

    # 设置标签和标题
    ax.set_xlabel('Feature Group', fontsize=13, fontweight='bold')
    ax.set_ylabel('Permutation Importance', fontsize=13, fontweight='bold')
    ax.set_title('Feature Group Permutation Importance',
                 fontsize=15, fontweight='bold', pad=20)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(group_names, rotation=45, ha='right', fontsize=11)

    # 添加网格线
    ax.grid(axis='y', alpha=0.3, linestyle='--', zorder=0)
    ax.set_axisbelow(True)

    # 在柱子上方显示数值
    for i, (bar, mean, std) in enumerate(zip(bars, means, stds)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2., height + std,
                f'{mean:.4f}',
                ha='center', va='bottom', fontsize=9, fontweight='bold')

    # 添加零基线
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8, alpha=0.5)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    return None


def main(args):
    feature_dir = args.feature_dir
    output_path = args.output_path
    cv_matrix = load_cv_matrix(feature_dir)
    model_list, X_test_list, y_test_list, test_index_list, feature_list = get_model_list(feature_dir, cv_matrix)
    prob_df = get_fold_proba_matrix(model_list, X_test_list, y_test_list, test_index_list, cv_matrix)
    prob_df_path = os.path.join(output_path, "prob_df.tsv")
    prob_df.to_csv(prob_df_path, sep="\t", index=False)
    perm_results = permutation_importance(
        model_list, feature_list, X_test_list, y_test_list,
        scoring="roc_auc", n_repeats=50
    )
    perm_tsv_path = os.path.join(output_path, "perm_results.tsv")
    perm_summary = pd.DataFrame([
        {"feature_group": name, "mean": float(v["mean"]), "std": float(v["std"])}
        for name, v in perm_results.items()
    ])
    perm_summary.to_csv(perm_tsv_path, sep="\t", index=False)
    permutation_importance_plot_path = os.path.join(output_path, "permutation_importance.png")
    plot_permutation_importance(perm_results, permutation_importance_plot_path)
    auc_plot_path = os.path.join(output_path, "auc.png")
    get_auc_plot(model_list, X_test_list, y_test_list, auc_plot_path)
    feature_importance_boxplot_path = os.path.join(output_path, "feature_importance_boxplot.png")
    feature_importance_boxplot(model_list, feature_list, feature_importance_boxplot_path)
    return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--feature_dir", type=str, required=True, help="path to feature directory")
    parser.add_argument("-o", "--output_path", type=str, required=True, help="path to output directory")
    args = parser.parse_args()
    main(args)
