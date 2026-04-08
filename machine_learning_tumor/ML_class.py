import os
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from flaml import AutoML
from sklearn import set_config
from imblearn.pipeline import Pipeline
# from sklearn.pipeline import Pipeline
# from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import RepeatedStratifiedKFold
from pyHSICLasso import HSICLasso
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import GenericUnivariateSelect, mutual_info_classif
import joblib
import optuna
import argparse
from typing import Tuple


set_config(display="diagram")


class HSICLassoTransformer(HSICLasso, BaseEstimator, TransformerMixin):
    """
    多重继承，sklearn/Transformer+HSICLasso。
    这样就能在使用Pipeline时调用 fit / transform，
    同时也能直接调用 HSICLasso 里已有的方法（dump, get_index, plot_path 等）。
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
        这里把HSICLasso相关的主要参数都列出来，
        让HSICLassoTransformer在实例化时可设置。
        """
        # 先调用父类 HSICLasso 的初始化，确保其内部属性都准备好了
        HSICLasso.__init__(self)

        # 下面这些是包装器本身需要的属性
        self.mode = mode
        self.num_feat = num_feat
        self.B = B
        self.M = M
        self.discrete_x = discrete_x
        self.max_neighbors = max_neighbors
        self.n_jobs = n_jobs
        self.covars = covars
        self.covars_kernel = covars_kernel
        self.feature_importances_ = None    # 存放输入特征的重要性
        self.selected_idx_ = None  # 用于在 transform() 阶段知道选了哪些列
        self.n_features_ = None  # 记录fit时的特征总数

    def fit(self, X, y=None):
        """
        符合 sklearn fit() 规范：
        调用classification或regression
        记录选出的特征索引
        记录[输入特征]的重要性
        """
        X = np.asarray(X)
        n_samples, n_features = X.shape
        
        # 记录特征数量
        self.n_features_ = n_features

        if y is None:
            raise ValueError("HSICLasso需要监督信息，y不可为空。")
        y = np.asarray(y).ravel()

        X_in, Y_in = X, y
        self.input(X_in, Y_in)   # 调用HSICLasso的 input 方法

        # 根据 mode 调用 HSICLasso 的分类或回归
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
            raise ValueError("mode必须是 'classification' 或 'regression'。")

        # -------------- 3) 记录选中列索引 ---------------
        # HSICLasso 会把最终选出的特征存到 self.A
        # self.A => (selected_features, ) (索引)
        # 由于行=特征，所以 self.A 正好对应 sklearn 里的特征列
        if not hasattr(self, 'A') or self.A is None:
            raise RuntimeError("HSICLasso 未能成功选择特征，self.A 不存在")
        
        self.selected_idx_ = self.A

        # -------------- 4) 计算并存储 feature_importances_ ---------------
        # 做法：创建一个长度 n_features 的 0 数组，对选中的特征给出非0分数
        # 例如 (beta[i][0] / maxval) 来自 HSICLasso dump() 的思路
        feature_importances = np.zeros(n_features)

        # 如果 self.beta 存在、且 A 非空，就可做标准化
        # 例如 dump() 里: maxval = self.beta[self.A[0]][0]
        # 注意 self.beta 是一个2D数组: shape(#features, ?)，看它实际情况(跟选择的邻近neighbor数目有关)
        if hasattr(self, 'A') and len(self.A) > 0 and hasattr(self, 'beta') and self.beta is not None:
            maxval = float(self.beta[self.A[0]][0])  # 取第一个被选中特征的 beta，beta降序排列，第一个特征beta最大
            
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
        只保留HSICLasso选中的特征列
        """
        if self.selected_idx_ is None:
            raise RuntimeError("必须先调用fit()再transform()。")
        
        if len(self.selected_idx_) == 0:
            raise ValueError("没有选中任何特征，无法进行 transform")

        X = np.asarray(X)
        return X[:, self.selected_idx_]
    
    def fit_transform(self, X, y=None):
        """
        先 fit 再 transform
        """
        return self.fit(X, y).transform(X)

    def get_support(self, indices: bool = False):
        """
        indices=True，返回一个索引数组，比如 [0, 2, 5...].
        indices=False，返回一个长度为 n_features_ 的布尔数组，选中的特征对应 True。
        """
        if self.selected_idx_ is None:
            raise ValueError("HSICLassoTransformer还未fit，无法调用get_support")
        
        if self.n_features_ is None:
            raise ValueError("n_features_ 未设置，请先调用 fit()")

        if indices:
            return self.selected_idx_
        else:
            # 生成布尔mask
            mask = np.zeros(self.n_features_, dtype=bool)
            mask[self.selected_idx_] = True
            return mask


def train_model(X_train, y_train, automl_params, output_dir: str, data_type: str = "default", 
                n_jobs: int = -1, n_trials: int = 10):
    """
    Train model with optional feature selection based on data type
    
    Args:
        X_train: Training features
        y_train: Training labels
        automl_params: AutoML parameters
        output_dir: Output directory
        data_type: Type of data ('artemis', 'end_motif', or other)
        n_jobs: Number of parallel jobs for HSICLasso and Optuna
        n_trials: Number of Optuna trials
    """
    def objective(trial):
        num_feat = trial.suggest_int("num_feat", 5, 25)
        oversampler = SMOTE(random_state=42)
        automl = AutoML(**automl_params)
        
        if data_type == "artemis":
            hsiclasso = HSICLassoTransformer(num_feat=num_feat, n_jobs=n_jobs)
            pipeline = Pipeline([
                ("oversampler", oversampler),
                ("hsiclasso", hsiclasso),
                ("automl", automl),
            ])
            
        elif data_type == "end_motif":
            selector = GenericUnivariateSelect(
                mode="k_best",
                score_func=mutual_info_classif,
                param=num_feat
            )
            pipeline = Pipeline([
                ("oversampler", oversampler),
                ("selector", selector),
                ("automl", automl),
            ])
            
        else:
            selector = GenericUnivariateSelect(
                mode="k_best",
                score_func=mutual_info_classif,
                param=3000
            )
            hsiclasso = HSICLassoTransformer(num_feat=num_feat, n_jobs=n_jobs)
            pipeline = Pipeline([
                ("oversampler", oversampler),
                ("selector", selector),
                ("hsiclasso", hsiclasso),
                ("automl", automl),
            ])
        pipeline.fit(X_train, y_train)
        automl = pipeline.steps[-1][1]
        best_loss = automl.best_loss
        trial.set_user_attr("best_val_loss", best_loss)
        trial.set_user_attr("pipeline", pipeline)
        return best_loss
    
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials)
    best_val_loss = study.best_trial.user_attrs["best_val_loss"]
    best_pipeline = study.best_trial.user_attrs["pipeline"]
    automl = best_pipeline.steps[-1][1]
    best_params = {**(study.best_trial.params), **(automl.best_config)}
    best_estimator = automl.best_estimator
    joblib.dump(best_pipeline, f"{output_dir}/best_pipeline.pkl", compress=3, protocol=4)
    joblib.dump(best_params, f"{output_dir}/best_params.pkl", compress=3, protocol=4)
    val_accuracy = 1 - best_val_loss
    return best_pipeline, val_accuracy, best_estimator, best_params


def test_model(
    model,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    output_dir: str,
):  
    y_pred_test = model.predict(X_test)
    y_true_test = y_test
    test_accuracy = accuracy_score(y_true_test, y_pred_test)
    print(f"test accuracy: {test_accuracy}\n")
    return test_accuracy


def nestcv(
    X: pd.DataFrame,
    y: pd.Series,
    stage: np.ndarray,
    outer_cv: int,
    outer_repeats: int,
    automl_params: dict,
    output_dir: str,
    data_type: str = "default",
    n_jobs: int = -1,
    n_trials: int = 10,
):
    stage = np.asarray(stage)
    if len(stage) != len(X):
        raise ValueError("stage length must match X")
    os.makedirs(output_dir, exist_ok=True)
    cv = RepeatedStratifiedKFold(n_splits=outer_cv, n_repeats=outer_repeats, random_state=42)
    results_list = []
    for fold_idx, (train_index, test_index) in enumerate(cv.split(X, y)):
        print(f"\n{'='*50}")
        print(f"第 {fold_idx + 1} 折开始...")
        print(f"{'='*50}\n")
        fold_dir = os.path.join(output_dir, f"fold_{fold_idx}")
        os.makedirs(fold_dir, exist_ok=True)
        X_train, y_train = X.iloc[train_index], y.iloc[train_index]
        X_test, y_test = X.iloc[test_index], y.iloc[test_index]
        stage_train = stage[train_index]
        stage_test = stage[test_index]

        np.save(os.path.join(fold_dir, "X_train.npy"), X_train)
        np.save(os.path.join(fold_dir, "X_test.npy"), X_test)
        np.save(os.path.join(fold_dir, "y_train.npy"), y_train)
        np.save(os.path.join(fold_dir, "y_test.npy"), y_test)
        np.save(os.path.join(fold_dir, "stage_train.npy"), stage_train)
        np.save(os.path.join(fold_dir, "stage_test.npy"), stage_test)
        np.save(os.path.join(fold_dir, "train_index.npy"), train_index)
        np.save(os.path.join(fold_dir, "test_index.npy"), test_index)

        automl_params["log_file_name"] = f"{fold_dir}/automl.log"
        best_pipeline, val_accuracy, best_estimator, best_params = train_model(
            X_train, y_train, automl_params, fold_dir, data_type, n_jobs, n_trials
        )

        test_accuracy = test_model(
            best_pipeline, X_test, y_test, fold_dir
        )

        results_list.append({
            'fold': fold_idx,
            'best_val_accuracy': val_accuracy,
            'test_accuracy': test_accuracy,
            'best_estimator': best_estimator,
            'best_params': best_params,
        })

    results_df = pd.DataFrame(results_list)
    return results_df


def preprocess(
    vst_counts: pd.DataFrame,
    metadata: pd.DataFrame,
    label: str = "type",
):
    """
    Preprocess VST counts

    Args:
        vst_counts: pd.DataFrame with VST counts (n_samples, n_features)
                   index is sample, columns are feature names
        metadata: pd.DataFrame with metadata containing 'sample', label, and 'stage' columns
        label: Column name in metadata for grouping

    Returns:
        X: Feature data (n_samples, n_features)
        y: Labels (n_samples,)
        stage: pd.Series with index = X.index, stage values from metadata
    """
    if "stage" not in metadata.columns:
        raise ValueError("metadata must contain 'stage' column")
    X = vst_counts.copy()

    X["sample"] = X.index
    groups = dict(zip(metadata["sample"].astype(str), metadata[label]))
    X["group"] = X["sample"].astype(str).map(groups)
    stage_map = dict(zip(metadata["sample"].astype(str), metadata["stage"]))
    X["stage"] = X["sample"].astype(str).map(stage_map)
    X = X.dropna(subset=["group"])
    y = pd.Series(X["group"].values.astype(float).astype(int), index=X.index)
    stage = X["stage"].copy()
    X = X.drop(columns=["sample", "group", "stage"])

    return X, y, stage


def filter_by_type1_annotation(
    X: pd.DataFrame,
    y: pd.Series,
    annotation_path: str,
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Restrict to samples with type1 in ['tumor', 'benign'] from sample_annotation.tsv.
    tumor -> label 1, benign -> label 0. Samples not in annotation or not in X are dropped.
    """
    ann = pd.read_csv(annotation_path, sep="\t")
    sub = ann[ann["type1"].isin(["tumor", "benign"])].copy()
    sub = sub.set_index("sample")
    keep = sub.index.intersection(X.index)
    if len(keep) == 0:
        raise ValueError("No overlap between annotation (tumor/benign) and X.index")
    return X.loc[keep], y.loc[keep]


def main(args):
    vst_counts = pd.read_csv(args.vst_counts, sep="\t", index_col=0)
    metadata = pd.read_csv(args.metadata)

    # Preprocess data
    X, y, stage = preprocess(
        vst_counts=vst_counts,
        metadata=metadata,
        label=args.label
    )

    script_dir = os.path.dirname(os.path.abspath(__file__))
    annotation_path = os.path.join(script_dir, "sample_annotation.tsv")
    if not os.path.isfile(annotation_path):
        raise FileNotFoundError(f"sample_annotation.tsv not found at {annotation_path}")
    X, y = filter_by_type1_annotation(X, y, annotation_path)
    stage = stage.loc[X.index].values
    print(f"Filtered to type1 in (tumor, benign): n_samples={len(X)}, tumor={int((y == 1).sum())}, benign={int((y == 0).sum())}")

    cv = RepeatedStratifiedKFold(n_splits=args.inner_cv, n_repeats=args.inner_repeats, random_state=42)
    automl_params = {
        "task": "classification",
        "eval_method": "cv",
        "estimator_list": ["lgbm", "xgb_limitdepth", "rf", "extra_tree"],
        "metric": "accuracy",
        "n_jobs": args.automl_n_jobs,
        "time_budget": -1,
        "model_history": False,
        "split_type": cv,
        "max_iter": args.max_iter,
        "retrain_full": False,
    }
    results_df = nestcv(X, y, stage, args.outer_cv, args.outer_repeats, automl_params, args.output_dir,
                        args.data_type, args.n_jobs, args.n_trials)
    results_df.to_csv(os.path.join(args.output_dir, "nestcv_results.csv"), index=False)
    # automl = pipeline.steps[-1][1]
    # automl.config_history
    # automl.model
    # automl.best_estimator
    # automl.best_config
    # automl.best_model_for_estimator
    # automl.best_config_per_estimator
    # automl.best_loss_per_estimator
    # automl.best_loss
    # automl.best_result
    # automl.feature_transformer
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--vst_counts", type=str, required=True,
                        help="VST counts file path (format: n_samples, n_features, index is sample, columns are feature names)")
    parser.add_argument("-m", "--metadata", type=str, required=True, help="Metadata file path")
    parser.add_argument("-l", "--label", type=str, default="type", help="Label column name in metadata")
    parser.add_argument("-o", "--output_dir", type=str, required=True, help="Output directory")
    parser.add_argument("-t", "--data_type", type=str, required=True,
                        choices=["artemis", "end_motif", "FSD", "gene_counts", "consensus_peak", "OCR", "window", "default"],
                        help="Data type: 'artemis' (no selector), 'end_motif' (no hsiclasso), or others (use both)")
    
    # Parallelization parameters
    parser.add_argument("--n_jobs", type=int, default=-1, 
                        help="Number of parallel jobs for HSICLasso (-1 uses all cores)")
    parser.add_argument("--automl_n_jobs", type=int, default=-1,
                        help="Number of parallel jobs for AutoML (-1 uses all cores)")
    parser.add_argument("--n_trials", type=int, default=10,
                        help="Number of Optuna hyperparameter optimization trials")
    
    # Cross-validation parameters
    parser.add_argument("--inner_cv", type=int, default=5, help="Number of inner CV folds")
    parser.add_argument("--inner_repeats", type=int, default=3, help="Number of inner CV repeats")
    parser.add_argument("--outer_cv", type=int, default=5, help="Number of outer CV folds")
    parser.add_argument("--outer_repeats", type=int, default=2, help="Number of outer CV repeats")
    
    # AutoML parameters
    parser.add_argument("--max_iter", type=int, default=50, help="Maximum iterations for AutoML")
    
    args = parser.parse_args()
    main(args)