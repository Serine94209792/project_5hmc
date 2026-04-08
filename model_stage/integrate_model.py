import os
import numpy as np
import pandas as pd
from typing import Optional
import joblib
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.base import BaseEstimator, TransformerMixin
from pyHSICLasso import HSICLasso

# Class colors shared by plot_proba_boxplot and plot_multi_feature_boxplot (class 0, 1, 2)
CLASS_COLORS = ["#348ABD", "#E24A33", "#2ecc71"]


class IdentityTransformer(BaseEstimator, TransformerMixin):
    """
    当特征数量少于num_feat时，不使用HSICLassoTransformer，而是使用这个恒等变换器
    """
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        return X
    
    def fit_transform(self, X, y=None):
        return X


class HSICLassoTransformer(HSICLasso, BaseEstimator, TransformerMixin):
    """
    Multiple inheritance: sklearn Transformer + HSICLasso.
    Enables fit/transform in Pipeline while still using HSICLasso methods (dump, get_index, plot_path, etc.).
    """

    def __init__(self,
                 mode="classification",  # or "regression"
                 num_feat=20,
                 B=0,
                 M=1,
                 discrete_x=False,
                 max_neighbors=5,
                 n_jobs=-1,
                 covars=np.array([]),
                 covars_kernel="Gaussian"):
        """
        Expose main HSICLasso parameters for HSICLassoTransformer instantiation.
        """
        # Initialize parent HSICLasso first
        HSICLasso.__init__(self)

        # Wrapper attributes
        self.mode = mode
        self.num_feat = num_feat
        self.B = B
        self.M = M
        self.discrete_x = discrete_x
        self.max_neighbors = max_neighbors
        self.n_jobs = n_jobs
        self.covars = covars
        self.covars_kernel = covars_kernel
        self.feature_importances_ = None    # feature importance from input
        self.selected_idx_ = None  # selected column indices for transform()
        self.n_features_ = None  # number of features at fit time

    def fit(self, X, y=None):
        """
        Sklearn-style fit: run classification or regression, record selected indices and feature importances.
        """
        X = np.asarray(X)
        n_samples, n_features = X.shape
        
        self.n_features_ = n_features

        if y is None:
            raise ValueError("HSICLasso requires supervised target y.")
        y = np.asarray(y).ravel()

        X_in, Y_in = X, y
        self.input(X_in, Y_in)

        if self.mode == "classification":
            self.classification(
                num_feat=self.num_feat, B=self.B, M=self.M,
                discrete_x=self.discrete_x, max_neighbors=self.max_neighbors,
                n_jobs=self.n_jobs, covars=self.covars, covars_kernel=self.covars_kernel
            )
        elif self.mode == "regression":
            self.regression(
                num_feat=self.num_feat, B=self.B, M=self.M,
                discrete_x=self.discrete_x, max_neighbors=self.max_neighbors,
                n_jobs=self.n_jobs, covars=self.covars, covars_kernel=self.covars_kernel
            )
        else:
            raise ValueError("mode must be 'classification' or 'regression'.")

        # Record selected column indices (HSICLasso stores them in self.A)
        if not hasattr(self, 'A') or self.A is None:
            raise RuntimeError("HSICLasso did not select features; self.A is missing.")
        
        self.selected_idx_ = self.A

        # Compute and store feature_importances_ (beta-based, same idea as HSICLasso dump())
        feature_importances = np.zeros(n_features)
        if hasattr(self, 'A') and len(self.A) > 0 and hasattr(self, 'beta') and self.beta is not None:
            maxval = float(self.beta[self.A[0]][0])  # first selected feature has largest beta
            
            if maxval != 0:
                for idx in self.A:
                    feature_importances[idx] = float(self.beta[idx][0]) / maxval
            else:
                for idx in self.A:
                    feature_importances[idx] = 1.0

        self.feature_importances_ = feature_importances

        return self

    def transform(self, X):
        """
        Return only the columns selected by HSICLasso.
        """
        if self.selected_idx_ is None:
            raise RuntimeError("Call fit() before transform().")
        
        if len(self.selected_idx_) == 0:
            raise ValueError("No features selected; cannot transform.")

        X = np.asarray(X)
        return X[:, self.selected_idx_]
    
    def fit_transform(self, X, y=None):
        """
        Fit then transform.
        """
        return self.fit(X, y).transform(X)

    def get_support(self, indices: bool = False):
        """
        indices=True: return index array (e.g. [0, 2, 5, ...]).
        indices=False: return boolean array of length n_features_, True for selected.
        """
        if self.selected_idx_ is None:
            raise ValueError("HSICLassoTransformer not fitted; cannot call get_support.")
        
        if self.n_features_ is None:
            raise ValueError("n_features_ not set; call fit() first.")

        if indices:
            return self.selected_idx_
        else:
            mask = np.zeros(self.n_features_, dtype=bool)
            mask[self.selected_idx_] = True
            return mask


def fold_probs(
    ml_results_dir: str,
    n_folds: Optional[int] = None
):
    """
    Load and aggregate cross-validation prediction results (multi-class: all proba columns).
    
    Args:
        ml_results_dir: Path to ML results dir (e.g. "gene_counts" or "gene_counts/ML_results")
        n_folds: Number of folds. If None, auto-detect from existing fold_* dirs.
    
    Returns:
        df: pandas.DataFrame with columns:
            - test_index: original sample index (position in filtered 3-class dataset)
            - y_test: true label (0, 1, 2)
            - proba_0, proba_1, proba_2: predicted probabilities per class
    
    Note: If a sample appears in multiple folds, its probabilities are averaged.
    """
    
    if n_folds is None:
        fold_dirs = []
        i = 0
        while True:
            fold_path = os.path.join(ml_results_dir, f"fold_{i}")
            if os.path.exists(fold_path):
                fold_dirs.append(i)
                i += 1
            else:
                break
        n_folds = len(fold_dirs)
        if n_folds == 0:
            raise ValueError(f"no fold found in {ml_results_dir}")
        print(f"detected {n_folds} folds")
    
    df_list = []
    
    for fold_idx in range(n_folds):
        fold_dir = os.path.join(ml_results_dir, f"fold_{fold_idx}")
        pipeline_path = os.path.join(fold_dir, "best_pipeline.pkl")
        test_index_path = os.path.join(fold_dir, "test_index.npy")
        y_test_path = os.path.join(fold_dir, "y_test.npy")
        X_test_path = os.path.join(fold_dir, "X_test.npy")
        
        if not all(os.path.exists(p) for p in [pipeline_path, test_index_path, y_test_path, X_test_path]):
            raise ValueError(f"fold {fold_idx} missing necessary files")
        
        pipeline = joblib.load(pipeline_path)
        test_indices = np.load(test_index_path)
        y_test = np.load(y_test_path)
        X_test = np.load(X_test_path)
        y_proba = pipeline.predict_proba(X_test)
        n_classes = y_proba.shape[1]
        
        fold_data = {
            'test_index': test_indices,
            'y_test': y_test,
        }
        for k in range(n_classes):
            fold_data[f'proba_{k}'] = y_proba[:, k]
        df_list.append(pd.DataFrame(fold_data))
    
    if len(df_list) == 0:
        raise ValueError("no samples read successfully")
    
    df = pd.concat(df_list, ignore_index=True)
    
    y_test_check = df.groupby('test_index')['y_test'].apply(lambda x: x.nunique() == 1)
    if not y_test_check.all():
        inconsistent = y_test_check[~y_test_check].index.tolist()
        raise ValueError(f"y_test values not consistent for test_indices: {inconsistent}")
    
    agg_dict = {'y_test': 'first'}
    proba_cols = [c for c in df.columns if c.startswith('proba_')]
    for col in proba_cols:
        agg_dict[col] = 'mean'
    grouped = df.groupby('test_index', as_index=False).agg(agg_dict)
    
    duplicate_count = (df.groupby('test_index').size() > 1).sum()
    if duplicate_count > 0:
        print(f"Found {duplicate_count} samples in multiple folds, averaged their probabilities")
    
    return grouped


def aggregate_fold_probs(prob_dfs: dict):
    """
    Merge prediction DataFrames from multiple features by test_index.
    
    Args:
        prob_dfs: dict, key=feature name, value=DataFrame from fold_probs.
    
    Returns:
        df: pandas.DataFrame with all feature probabilities, merged on test_index:
            - test_index, y_test, {feature_name}_proba_0, _proba_1, _proba_2, ...
    """
    
    if not prob_dfs or len(prob_dfs) == 0:
        raise ValueError("prob_dfs must be a non-empty dictionary")
    
    dfs = []
    feature_names_used = []
    
    for feature_name, df in prob_dfs.items():
        proba_cols = [col for col in df.columns if col.startswith('proba_')]
        rename_dict = {col: f"{feature_name}_{col}" for col in proba_cols}
        df_renamed = df.rename(columns=rename_dict)
        
        dfs.append(df_renamed)
        feature_names_used.append(feature_name)
    
    merged_df = dfs[0]
    
    for i in range(1, len(dfs)):
        proba_cols_to_merge = [col for col in dfs[i].columns if col.startswith(feature_names_used[i])]
        merged_df = merged_df.merge(
            dfs[i][['test_index'] + proba_cols_to_merge],
            on='test_index',
            how='inner'
        )
    
    merged_df = merged_df.sort_values('test_index').reset_index(drop=True)
    return merged_df


def plot_multi_feature_boxplot(
    merged_df: pd.DataFrame,
    output_path: str = "multi_feature_boxplot.png",
    figsize: tuple = (8, 8),
):
    """
    Faceted plot: rows = datatype, columns = prob_k (prob_0, prob_1, prob_2).
    Each cell has 3 side-by-side violin + box plots for true classes C0, C1, C2 (same CLASS_COLORS).
    """
    import re
    proba_cols_all = [col for col in merged_df.columns if re.match(r'.+_proba_\d+$', col)]
    if not proba_cols_all:
        raise ValueError("No {feature}_proba_{k} columns found")
    class_set = sorted({int(col.split('_proba_')[-1]) for col in proba_cols_all})
    feature_names = sorted({col.rsplit('_proba_', 1)[0] for col in proba_cols_all})
    n_feat = len(feature_names)

    # Long-format: each row = (sample, datatype, prob_type, true_class, probability)
    rows = []
    for _, row in merged_df.iterrows():
        true_class = row["y_test"]
        for feat in feature_names:
            for k in class_set:
                col = f"{feat}_proba_{k}"
                if col not in merged_df.columns:
                    continue
                val = row[col]
                if pd.isna(val):
                    continue
                rows.append({
                    "Datatype": feat,
                    "ProbType": f"prob_{k}",
                    "TrueClass": true_class,
                    "Probability": val,
                })
    plot_data = pd.DataFrame(rows)
    if plot_data.empty:
        raise ValueError("No data to plot")

    sns.set_style("whitegrid")
    fig, axes = plt.subplots(n_feat, 3, sharey=True, figsize=figsize)
    if n_feat == 1:
        axes = axes.reshape(1, -1)

    hue_order = class_set
    for i, feat in enumerate(feature_names):
        for j, prob_j in enumerate(class_set):
            ax = axes[i, j]
            subset = plot_data[
                (plot_data["Datatype"] == feat) & (plot_data["ProbType"] == f"prob_{prob_j}")
            ]
            if subset.empty:
                ax.set_visible(False)
                continue
            sns.violinplot(
                data=subset,
                x="TrueClass",
                y="Probability",
                order=hue_order,
                palette=CLASS_COLORS[: len(class_set)],
                alpha=0.3,
                inner=None,
                ax=ax,
            )
            sns.boxplot(
                data=subset,
                x="TrueClass",
                y="Probability",
                order=hue_order,
                palette=CLASS_COLORS[: len(class_set)],
                width=0.5,
                linewidth=2,
                showfliers=False,
                ax=ax,
            )
            ax.set_ylim(-0.05, 1.05)
            ax.yaxis.grid(True, alpha=0.3)
            ax.set_axisbelow(True)
            ax.set_xlabel("True class" if i == n_feat - 1 else "")
            if j == 0:
                ax.set_ylabel(f"{feat}\nProbability", fontsize=10, fontweight="bold")
            else:
                ax.set_ylabel("")
                ax.set_yticklabels([])
            if i == 0:
                ax.set_title(f"prob_{prob_j}", fontsize=12, fontweight="bold")
            ax.set_xticklabels([f"C{c}" for c in hue_order])

    fig.suptitle("Prediction probability by datatype and prob type (True class in each cell)", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Multi-feature boxplot saved to {output_path}")
    return None


def main(args):
    feature_dirs = args.feature_dirs
    output_path = args.output_path
    os.makedirs(output_path, exist_ok=True)
    prob_dfs = {}
    for feature_dir in feature_dirs:
        ml_results_dir = feature_dir
        prob_df = fold_probs(ml_results_dir)
        feature_name = os.path.basename(os.path.normpath(feature_dir))
        prob_dfs[feature_name] = prob_df

    aggregate_df = aggregate_fold_probs(prob_dfs=prob_dfs)
    aggregate_df.to_csv(os.path.join(output_path, "aggregate_df.tsv"), sep="\t", index=False)
    multi_feature_plot = os.path.join(output_path, "multi_feature_boxplot.png")
    plot_multi_feature_boxplot(aggregate_df, multi_feature_plot)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--feature_dirs", type=str, nargs='+', required=True, help="path to feature directories")
    parser.add_argument("-o", "--output_path", type=str, required=True, help="path to output directory")
    args = parser.parse_args()
    main(args)
