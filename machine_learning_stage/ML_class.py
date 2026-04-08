import os
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from flaml import AutoML
from sklearn import set_config
from sklearn.pipeline import Pipeline
from sklearn.model_selection import RepeatedStratifiedKFold
from pyHSICLasso import HSICLasso
from sklearn.metrics import accuracy_score, log_loss
from sklearn.decomposition import PCA
from sklearn.preprocessing import FunctionTransformer, LabelEncoder
import joblib
import optuna
import argparse


set_config(display="diagram")


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


def train_model(X_train, y_train, automl_params, output_dir: str, use_hsiclasso: bool = True, num_feat: int = 80, n_trials: int = 10, hsiclasso_n_jobs: int = -1):
    def objective(trial):
        n_components = trial.suggest_int("pca_n_components", 10, 30)
        
        if use_hsiclasso:
            hsiclasso = HSICLassoTransformer(num_feat=num_feat, n_jobs=hsiclasso_n_jobs)
            X_selected = hsiclasso.fit_transform(X_train, y_train)
        else:
            X_selected = X_train

        max_comp = min(X_selected.shape[0], X_selected.shape[1])
        if n_components > max_comp:
            n_components = max(1, max_comp)
        if n_components < 1:
            raise ValueError("n_components must be at least 1")

        pca_temp = PCA(n_components=n_components)
        pca_temp.fit(X_selected)
        cumsum_variance = np.cumsum(pca_temp.explained_variance_ratio_)
        cumulative_variance = cumsum_variance[-1]  # Total cumulative variance for n_components
        
        print(f"Trial {trial.number}: n_components={n_components}, cumulative_variance={cumulative_variance:.4f}, use_hsiclasso={use_hsiclasso}")
        
        if use_hsiclasso:
            feature_selector = HSICLassoTransformer(num_feat=num_feat, n_jobs=hsiclasso_n_jobs)
        else:
            feature_selector = IdentityTransformer()
        
        pca = PCA(n_components=n_components)
        
        automl = AutoML(
            **automl_params,
        )
        pipeline = Pipeline([
            ("feature_selector", feature_selector),
            ("pca", pca),
            ("automl", automl),
        ])
        pipeline.fit(X_train, y_train)
        automl = pipeline.steps[-1][1]
        best_loss = automl.best_loss
        trial.set_user_attr("best_val_loss", best_loss)
        trial.set_user_attr("pipeline", pipeline)
        trial.set_user_attr("cumulative_variance", cumulative_variance)
        return best_loss
    
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials)
    best_val_loss = study.best_trial.user_attrs["best_val_loss"]
    best_pipeline = study.best_trial.user_attrs["pipeline"]
    cumulative_variance = study.best_trial.user_attrs["cumulative_variance"]
    automl = best_pipeline.steps[-1][1]
    best_params = {
        **(study.best_trial.params),
        "cumulative_variance": cumulative_variance,
        **(automl.best_config)
    }
    best_estimator = automl.best_estimator
    joblib.dump(best_pipeline, f"{output_dir}/best_pipeline.pkl", compress=3, protocol=4)
    joblib.dump(best_params, f"{output_dir}/best_params.pkl", compress=3, protocol=4)
    return best_pipeline, best_val_loss, best_estimator, best_params


def test_model(
    model,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    output_dir: str,
):
    y_pred_test = model.predict(X_test)
    y_prob = model.predict_proba(X_test)
    test_accuracy = accuracy_score(y_test, y_pred_test)
    test_log_loss = log_loss(y_test, y_prob)
    print(f"test accuracy: {test_accuracy}, test_log_loss: {test_log_loss}\n")
    return test_accuracy, test_log_loss


def nestcv(
    X: pd.DataFrame,
    y: pd.Series,
    outer_cv: int,
    outer_repeats: int,
    automl_params: dict,
    output_dir: str,
    num_feat: int = 80,
    n_trials: int = 10,
    hsiclasso_n_jobs: int = -1,
):  
    os.makedirs(output_dir, exist_ok=True)
    
    n_features = X.shape[1]
    use_hsiclasso = n_features >= num_feat
    
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
        
        np.save(os.path.join(fold_dir, "X_train.npy"), X_train)
        np.save(os.path.join(fold_dir, "X_test.npy"), X_test)
        np.save(os.path.join(fold_dir, "y_train.npy"), y_train)
        np.save(os.path.join(fold_dir, "y_test.npy"), y_test)
        np.save(os.path.join(fold_dir, "train_index.npy"), train_index)
        np.save(os.path.join(fold_dir, "test_index.npy"), test_index)
        
        automl_params["log_file_name"] = f"{fold_dir}/automl.log"
        best_pipeline, val_log_loss, best_estimator, best_params = train_model(
            X_train, y_train, automl_params, fold_dir,
            use_hsiclasso=use_hsiclasso, num_feat=num_feat, n_trials=n_trials, hsiclasso_n_jobs=hsiclasso_n_jobs
        )
        
        test_accuracy, test_log_loss = test_model(
            best_pipeline, X_test, y_test, fold_dir
        )
        
        results_list.append({
            'fold': fold_idx,
            'val_log_loss': val_log_loss,
            'test_accuracy': test_accuracy,
            'test_log_loss': test_log_loss,
            'best_estimator': best_estimator,
            'best_params': best_params,
        })
            
    results_df = pd.DataFrame(results_list)
    return results_df

