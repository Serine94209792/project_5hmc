import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import joblib
from sklearn.metrics import auc, precision_recall_curve, roc_auc_score, roc_curve
import argparse
from pathlib import Path


def _pr_auc_macro_ovr(y_test, y_prob):
    """Per-class One-vs-Rest PR-AUC, then macro average."""
    n_classes = y_prob.shape[1]
    pr_aucs = []
    for k in range(n_classes):
        y_binary = (y_test == k).astype(int)
        if y_binary.sum() == 0 or y_binary.sum() == len(y_binary):
            pr_aucs.append(0.0)
            continue
        prec, rec, _ = precision_recall_curve(y_binary, y_prob[:, k])
        pr_aucs.append(auc(rec, prec))
    return np.mean(pr_aucs)


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
    计算分组排列重要性（支持多分类：macro ROC-AUC，或 per-class OvR PR-AUC 再 macro 平均）
    """
    np.random.seed(random_state)

    y_prob = model.predict_proba(X_test)
    if scoring == "roc_auc":
        baseline_score = roc_auc_score(y_test, y_prob, multi_class='ovr', average='macro')
    elif scoring == "pr_auc":
        baseline_score = _pr_auc_macro_ovr(y_test, y_prob)
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

            y_prob_permuted = model.predict_proba(X_permuted)
            if scoring == "roc_auc":
                permuted_score = roc_auc_score(y_test, y_prob_permuted, multi_class='ovr', average='macro')
            elif scoring == "pr_auc":
                permuted_score = _pr_auc_macro_ovr(y_test, y_prob_permuted)

            group_scores.append(baseline_score - permuted_score)

        results[group_name] = {
            'importance': group_scores,
            'baseline': baseline_score
        }

    return results


def parse_feature_list(feature_list):
    """
    解析特征列表，支持新特征命名：
      - {datatype}_pseudotime
      - {datatype}_score_{k}
    """
    import re

    feature_info = []
    for idx, feat_name in enumerate(feature_list):
        feat_str = str(feat_name)
        feat_type: str
        class_label: str = ""

        if feat_str.endswith("_pseudotime"):
            feat_type = feat_str[: -len("_pseudotime")]
        else:
            m = re.match(r"^(.+)_score_(\d+)$", feat_str)
            if m is None:
                # Fallback: keep whole string as feature_type so permutation works
                feat_type = feat_str
            else:
                feat_type = m.group(1)
                class_label = m.group(2)

        feature_info.append(
            {
                "index": idx,
                "feature_name": feat_str,
                "feature_type": feat_type,
                "class": class_label,
            }
        )
    return pd.DataFrame(feature_info)


def feature_importance_boxplot(
    model_list,
    feature_list,
    output_path
):
    """
    绘制模型特征系数的箱线图（多分类：3 个子图，每个类别一个）
    coef_.shape = (n_classes, n_features)
    """
    feature_info_df = parse_feature_list(feature_list)
    feature_types = feature_info_df['feature_type'].unique()
    n_classes = model_list[0].coef_.shape[0]
    n_models = len(model_list)

    cmap = plt.get_cmap('tab10')
    colors = [cmap(i) for i in range(len(feature_types))]

    fig, axes = plt.subplots(1, n_classes, figsize=(6 * n_classes, 6))
    if n_classes == 1:
        axes = [axes]

    for class_k in range(n_classes):
        ax = axes[class_k]
        coef_matrix = np.array([np.abs(model.coef_[class_k]) for model in model_list])
        df = pd.concat([feature_info_df] * n_models, ignore_index=True)
        df['coef'] = coef_matrix.T.ravel()

        positions = []
        data_to_plot = []
        labels = []
        for i, feat_type in enumerate(feature_types):
            mask = df['feature_type'] == feat_type
            combined_data = df[mask]['coef'].values
            positions.append(i)
            data_to_plot.append(combined_data)
            labels.append(feat_type)

        bp = ax.boxplot(data_to_plot, positions=positions, widths=0.6,
                        patch_artist=True, showfliers=False,
                        boxprops=dict(linewidth=1.5),
                        medianprops=dict(color='black', linewidth=2),
                        whiskerprops=dict(linewidth=1.5),
                        capprops=dict(linewidth=1.5))
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        for i, (pos, data) in enumerate(zip(positions, data_to_plot)):
            jitter = np.random.normal(0, 0.04, size=len(data))
            x_data = np.ones(len(data)) * pos + jitter
            ax.scatter(x_data, data, alpha=0.4, s=20, color=colors[i], edgecolors='none')

        ax.set_xticks(positions)
        ax.set_xticklabels(labels, fontsize=11, rotation=45, ha='right')
        ax.set_ylabel('|Coefficient|', fontsize=12, fontweight='bold')
        ax.set_xlabel('Feature Type', fontsize=12, fontweight='bold')
        ax.set_title(f'Class {class_k} (1/2/3/4)', fontsize=14, fontweight='bold', pad=20)
        ax.grid(axis='y', alpha=0.3, linestyle='--')

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
    n_folds: int | None = None,
):
    model_list = []
    X_test_list = []
    y_test_list = []
    test_index_list = []
    ml_results_dir = os.path.join(feature_dir, "ML_results")
    if n_folds is None:
        n_folds = infer_ml_result_fold_count(feature_dir)
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

    non_feature_candidates = ["test_index", "y_test", "sample", "true_state"]
    non_feature_cols = [c for c in non_feature_candidates if c in cv_matrix.columns]
    cv_matrix_dropped = cv_matrix.drop(columns=non_feature_cols)
    fallback_feature_list = cv_matrix_dropped.columns.tolist()
    feature_names_path = os.path.join(ml_results_dir, "fold_0", "feature_names.npy")
    if os.path.isfile(feature_names_path):
        feature_list = np.load(feature_names_path, allow_pickle=True).tolist()
    else:
        n_feat = X_test_list[0].shape[1]
        feature_list = fallback_feature_list[:n_feat] if len(fallback_feature_list) >= n_feat else fallback_feature_list
    return model_list, X_test_list, y_test_list, test_index_list, feature_list


def infer_ml_result_fold_count(feature_dir: str) -> int:
    ml_results_dir = Path(feature_dir) / "ML_results"
    nestcv_results_path = ml_results_dir / "nestcv_results.csv"
    if nestcv_results_path.is_file():
        results_df = pd.read_csv(nestcv_results_path)
        if "fold" in results_df.columns and len(results_df) > 0:
            return int(results_df["fold"].max()) + 1
        if len(results_df) > 0:
            return int(len(results_df))

    fold_dirs = sorted(
        p for p in ml_results_dir.iterdir()
        if p.is_dir() and p.name.startswith("fold_")
    ) if ml_results_dir.is_dir() else []
    if fold_dirs:
        return len(fold_dirs)

    raise FileNotFoundError(
        f"Cannot infer fold count from {ml_results_dir}: "
        "missing nestcv_results.csv and fold_* directories."
    )


def get_fold_proba_matrix(
    model_list: list,
    X_test_list: list,
    y_test_list: list,
    test_index_list: list,
    cv_matrix: pd.DataFrame,
    n_repeats: int = 3,
) -> pd.DataFrame:
    n_folds = len(model_list)
    # If the repeats can't be inferred (e.g. cv_repeats mismatch), fall back to n_repeats=1.
    if n_repeats < 1:
        n_repeats = 1
    if n_folds % n_repeats != 0:
        raise ValueError(
            f"n_folds ({n_folds}) must be divisible by n_repeats ({n_repeats})"
        )
    folds_per_repeat = n_folds // n_repeats
    n_classes = model_list[0].predict_proba(X_test_list[0]).shape[1]

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
        proba_list = [
            model_list[i].predict_proba(X_test_list[i])
            for i in range(start_fold, end_fold)
        ]
        proba_r = np.concatenate(proba_list, axis=0)
        sort_idx = np.argsort(test_index_r)
        test_index_r = test_index_r[sort_idx]
        proba_r = proba_r[sort_idx]
        if result is None:
            y_test_r = y_test_r[sort_idx]
            result = pd.DataFrame({
                "test_index": test_index_r,
                "y_test": y_test_r,
            })
            for k in range(n_classes):
                result[f"prob_{k}_repeat_{r}"] = proba_r[:, k]
        else:
            for k in range(n_classes):
                result[f"prob_{k}_repeat_{r}"] = proba_r[:, k]
    result = result.reset_index(drop=True)
    return result


def get_auc_plot(
    model_list,
    X_test_list,
    y_test_list,
    output_path
):
    """
    绘制多分类 ROC 曲线：每个类别 OvR 一条曲线，标注 macro AUC。
    """
    mean_fpr = np.linspace(0, 1, 100)
    n_classes = model_list[0].predict_proba(X_test_list[0]).shape[1]
    class_names = [f'Class {k}' for k in range(n_classes)]

    all_roc_aucs = []
    tprs_per_class = {k: [] for k in range(n_classes)}

    for model, X_test, y_test in zip(model_list, X_test_list, y_test_list):
        y_prob = model.predict_proba(X_test)
        micro_roc = roc_auc_score(y_test, y_prob, multi_class='ovr', average='micro')
        macro_roc = roc_auc_score(y_test, y_prob, multi_class='ovr', average='macro')
        all_roc_aucs.append((micro_roc, macro_roc))

        for k in range(n_classes):
            y_bin = (y_test == k).astype(int)
            if y_bin.sum() == 0 or y_bin.sum() == len(y_bin):
                tprs_per_class[k].append(np.interp(mean_fpr, [0, 1], [0, 1]))
                continue
            fpr, tpr, _ = roc_curve(y_bin, y_prob[:, k])
            tprs_per_class[k].append(np.interp(mean_fpr, fpr, tpr))

    mean_auc_roc_micro = np.mean([x[0] for x in all_roc_aucs])
    std_auc_roc_micro = np.std([x[0] for x in all_roc_aucs])
    mean_auc_roc_macro = np.mean([x[1] for x in all_roc_aucs])
    std_auc_roc_macro = np.std([x[1] for x in all_roc_aucs])

    per_class_roc_auc = []
    for k in range(n_classes):
        mean_tpr_k = np.mean(tprs_per_class[k], axis=0)
        mean_tpr_k[-1] = 1.0
        per_class_roc_auc.append(auc(mean_fpr, mean_tpr_k))

    fig, ax1 = plt.subplots(1, 1, figsize=(6, 6))
    colors = plt.cm.tab10(np.linspace(0, 1, n_classes))

    for k in range(n_classes):
        mean_tpr = np.mean(tprs_per_class[k], axis=0)
        mean_tpr[-1] = 1.0
        std_tpr = np.std(tprs_per_class[k], axis=0)
        ax1.plot(mean_fpr, mean_tpr, color=colors[k], linewidth=2,
                 label=f'{class_names[k]} (AUC={per_class_roc_auc[k]:.3f})')
        ax1.fill_between(mean_fpr, mean_tpr - std_tpr, mean_tpr + std_tpr, color=colors[k], alpha=0.2)
    ax1.plot([0, 1], [0, 1], 'k--', linewidth=1.5, alpha=0.5, label='Random')
    ax1.plot([], [], ' ', label=f'Macro AUC={mean_auc_roc_macro:.3f}')
    ax1.plot([], [], ' ', label=f'Micro AUC={mean_auc_roc_micro:.3f}')
    ax1.set_xlabel('False Positive Rate', fontsize=12, fontweight='bold')
    ax1.set_ylabel('True Positive Rate', fontsize=12, fontweight='bold')
    ax1.set_title('ROC', fontsize=11, fontweight='bold', pad=15)
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
            'tprs_per_class': tprs_per_class,
            'mean_auc_micro': mean_auc_roc_micro,
            'std_auc_micro': std_auc_roc_micro,
            'mean_auc_macro': mean_auc_roc_macro,
            'std_auc_macro': std_auc_roc_macro,
        },
    }
    data_path = os.path.join(output_dir, 'auc_plot_data.pkl')
    joblib.dump(plot_data, data_path, compress=3, protocol=4)
    return None


def reload_and_plot_auc(data_path, output_path):
    """
    从保存的数据重新绘制 ROC 曲线（支持多分类格式）。
    """
    plot_data = joblib.load(data_path)
    roc = plot_data['roc']
    mean_fpr = roc['mean_fpr']

    fig, ax1 = plt.subplots(1, 1, figsize=(6, 6))

    if 'tprs_per_class' in roc:
        n_classes = len(roc['tprs_per_class'])
        colors = plt.cm.tab10(np.linspace(0, 1, n_classes))
        for k in range(n_classes):
            mean_tpr = np.mean(roc['tprs_per_class'][k], axis=0)
            mean_tpr[-1] = 1.0
            std_tpr = np.std(roc['tprs_per_class'][k], axis=0)
            ax1.plot(mean_fpr, mean_tpr, color=colors[k], linewidth=2, label=f'Class {k}')
            ax1.fill_between(mean_fpr, mean_tpr - std_tpr, mean_tpr + std_tpr, color=colors[k], alpha=0.2)
        ax1.plot([0, 1], [0, 1], 'k--', linewidth=1.5, alpha=0.5, label='Random')
        ax1.set_title(f"ROC (micro = {roc['mean_auc_micro']:.3f} ± {roc['std_auc_micro']:.3f}, macro = {roc['mean_auc_macro']:.3f} ± {roc['std_auc_macro']:.3f})", fontsize=11, fontweight='bold', pad=15)
    else:
        mean_tpr = roc['mean_tpr']
        std_tpr = roc['std_tpr']
        ax1.plot(mean_fpr, mean_tpr, color='#FF6B6B', linewidth=2.5, label=f"ROC (AUC = {roc['mean_auc']:.3f} ± {roc['std_auc']:.3f})")
        ax1.fill_between(mean_fpr, mean_tpr - std_tpr, mean_tpr + std_tpr, color='#FF6B6B', alpha=0.2)
        ax1.plot([0, 1], [0, 1], 'k--', linewidth=1.5, alpha=0.5, label='Random')
        ax1.set_title('ROC Curve', fontsize=14, fontweight='bold', pad=15)

    ax1.set_xlabel('False Positive Rate', fontsize=12, fontweight='bold')
    ax1.set_ylabel('True Positive Rate', fontsize=12, fontweight='bold')
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
    n_folds = infer_ml_result_fold_count(feature_dir)
    model_list, X_test_list, y_test_list, test_index_list, feature_list = get_model_list(
        feature_dir,
        cv_matrix,
        n_folds=n_folds,
    )
    prob_df = get_fold_proba_matrix(
        model_list,
        X_test_list,
        y_test_list,
        test_index_list,
        cv_matrix,
        n_repeats=args.cv_repeats,
    )
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
    parser.add_argument("--cv_repeats", type=int, default=3, help="number of outer CV repeats used in ML_results")
    args = parser.parse_args()
    main(args)
