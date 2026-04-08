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


def load_metadata(metadata_path: str) -> pd.DataFrame:
    meta = pd.read_csv(metadata_path)
    return meta[["sample", "type", "stage"]].copy()


def process_metadata(meta: pd.DataFrame) -> pd.DataFrame:
    meta = meta.copy()
    meta.loc[meta["type"] == 0, "stage"] = 0
    stage = meta["stage"].astype(object)
    stage.loc[stage.isin([3, 4, 3.0, 4.0])] = "3/4"
    meta["stage"] = stage.astype(str)
    return meta


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
    Load and aggregate cross-validation prediction results.
    
    Args:
        ml_results_dir: Path to ML results dir (e.g. "gene_counts/ML_results")
        n_folds: Number of folds. If None, auto-detect from existing fold_* dirs.
    
    Returns:
        df: pandas.DataFrame with columns:
            - test_index: original sample index
            - y_test: true label
            - proba_1: predicted probability for class 1
    
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
        
        fold_data = {
            'test_index': test_indices, 
            'y_test': y_test,
            'proba_1': y_proba[:, 1]
        }
        df_list.append(pd.DataFrame(fold_data))
    
    if len(df_list) == 0:
        raise ValueError("no samples read successfully")
    
    df = pd.concat(df_list, ignore_index=True)
    
    y_test_check = df.groupby('test_index')['y_test'].apply(lambda x: x.nunique() == 1)
    if not y_test_check.all():
        inconsistent = y_test_check[~y_test_check].index.tolist()
        raise ValueError(f"y_test values not consistent for test_indices: {inconsistent}")
    
    grouped = df.groupby('test_index', as_index=False).agg({
        'y_test': 'first',
        'proba_1': 'mean'
    })
    
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
            - test_index, y_test, {feature_name}_proba_1
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


def add_stage(aggregate_df: pd.DataFrame, metadata: pd.DataFrame) -> pd.DataFrame:
    """
    Add stage to aggregate_df by test_index: metadata row i corresponds to sample index i.
    aggregate_df may have fewer rows than metadata (only samples that appear in all feature CVs).
    """
    meta_sub = metadata.iloc[aggregate_df["test_index"].values].reset_index(drop=True)
    aggregate_df = aggregate_df.copy()
    aggregate_df["stage"] = meta_sub["stage"].values
    return aggregate_df


def plot_multi_feature_boxplot(
    merged_df: pd.DataFrame,
    output_path: str = "multi_feature_boxplot.png",
    figsize: tuple = (12, 8),
    positive_class: int = 1
):
    """
    Boxplot comparison of prediction probability across multiple features.
    
    Args:
        merged_df: DataFrame from aggregate_fold_probs
        output_path: Output figure path
        figsize: Figure size
        positive_class: Positive class label
    """
    import seaborn as sns
    
    proba_cols = [col for col in merged_df.columns if col.endswith(f'_proba_{positive_class}')]
    
    if len(proba_cols) == 0:
        raise ValueError(f"No probability columns found for class {positive_class}")
    
    feature_names = [col.replace(f'_proba_{positive_class}', '') for col in proba_cols]
    
    # Long format for plotting
    plot_data_list = []
    for feature_name, proba_col in zip(feature_names, proba_cols):
        temp_df = merged_df[['y_test', proba_col]].copy()
        temp_df['Feature'] = feature_name
        temp_df['Probability'] = temp_df[proba_col]
        temp_df['Group'] = temp_df['y_test'].map({0: 'Negative', 1: 'Positive'})
        plot_data_list.append(temp_df[['Feature', 'Group', 'Probability']])
    
    plot_data = pd.concat(plot_data_list, ignore_index=True)
    
    sns.set_style("whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    for idx, group in enumerate(['Negative', 'Positive']):
        ax = axes[idx]
        group_data = plot_data[plot_data['Group'] == group]
        
        sns.boxplot(
            data=group_data,
            x='Feature',
            y='Probability',
            palette='Set2',
            width=0.6,
            linewidth=2,
            ax=ax
        )
        
        sns.violinplot(
            data=group_data,
            x='Feature',
            y='Probability',
            palette='Set2',
            alpha=0.3,
            inner=None,
            ax=ax
        )
        
        sns.stripplot(
            data=group_data,
            x='Feature',
            y='Probability',
            color='black',
            alpha=0.2,
            size=2,
            ax=ax
        )
        
        ax.set_xlabel('Feature', fontsize=12, fontweight='bold')
        ax.set_ylabel('Predicted Probability', fontsize=12, fontweight='bold')
        ax.set_title(f'{group} Samples (y_test={idx})', fontsize=14, fontweight='bold', pad=15)
        ax.tick_params(axis='x', rotation=45, labelsize=10)
        ax.yaxis.grid(True, alpha=0.3)
        ax.set_axisbelow(True)
    
    plt.suptitle(f'Prediction Probability Distribution by Feature (Class {positive_class})',
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Multi-feature boxplot saved to {output_path}")
    
    stats_summary = plot_data.groupby(['Feature', 'Group'])['Probability'].agg(['median', 'std', 'count'])
    return stats_summary


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
    metadata = load_metadata(args.metadata_path)
    metadata = process_metadata(metadata)
    aggregate_df = add_stage(aggregate_df, metadata)
    aggregate_df.to_csv(os.path.join(output_path, "aggregate_df.tsv"), sep="\t", index=False)
    multi_feature_plot = os.path.join(output_path, "multi_feature_boxplot.png")
    plot_multi_feature_boxplot(aggregate_df, multi_feature_plot)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--feature_dirs", type=str, nargs='+', required=True, help="path to feature directories")
    parser.add_argument("-o", "--output_path", type=str, required=True, help="path to output directory")
    parser.add_argument("-m", "--metadata_path", type=str, required=True, help="path to metadata CSV (e.g. cfDNA_metadata2_TNM.csv)")
    args = parser.parse_args()
    main(args)
