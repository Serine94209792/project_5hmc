import argparse
import os
import random
from typing import Any, Dict, List, Tuple
import joblib
import numpy as np
import optuna
import pandas as pd
import torch
from pyHSICLasso import HSICLasso
from scipy.stats import mannwhitneyu
from sklearn.base import BaseEstimator, RegressorMixin, TransformerMixin
from sklearn.feature_selection import mutual_info_classif
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedShuffleSplit
from sklearn.pipeline import Pipeline

HSICLASSO_N_FEATURES_MIN = 30
HSICLASSO_N_FEATURES_MAX = 100


class HSICLassoTransformer(HSICLasso, BaseEstimator, TransformerMixin):
    """sklearn-compatible HSIC-Lasso feature selector."""

    def __init__(
        self,
        mode: str = "regression",
        num_feat: int = 80,
        B: int = 0,
        M: int = 1,
        discrete_x: bool = False,
        max_neighbors: int = 5,
        n_jobs: int = -1,
        covars: np.ndarray = np.array([]),
        covars_kernel: str = "Gaussian",
    ):
        HSICLasso.__init__(self)
        self.mode = mode
        self.num_feat = num_feat
        self.B = B
        self.M = M
        self.discrete_x = discrete_x
        self.max_neighbors = max_neighbors
        self.n_jobs = n_jobs
        self.covars = covars
        self.covars_kernel = covars_kernel
        self.selected_idx_: np.ndarray | None = None
        self.n_features_: int | None = None

    def fit(self, X, y=None):
        X = np.asarray(X)
        if y is None:
            raise ValueError("HSICLassoTransformer requires y.")
        y = np.asarray(y).ravel()
        self.n_features_ = X.shape[1]

        if self.num_feat >= self.n_features_:
            raise ValueError(
                f"hsiclasso_n_features must be < input feature count. "
                f"Got num_feat={self.num_feat}, input_features={self.n_features_}."
            )

        self.input(X, y)
        if self.mode == "classification":
            self.classification(
                num_feat=self.num_feat,
                B=self.B,
                M=self.M,
                discrete_x=self.discrete_x,
                max_neighbors=self.max_neighbors,
                n_jobs=self.n_jobs,
                covars=self.covars,
                covars_kernel=self.covars_kernel,
            )
        else:
            self.regression(
                num_feat=self.num_feat,
                B=self.B,
                M=self.M,
                discrete_x=self.discrete_x,
                max_neighbors=self.max_neighbors,
                n_jobs=self.n_jobs,
                covars=self.covars,
                covars_kernel=self.covars_kernel,
            )
        if not hasattr(self, "A") or self.A is None or len(self.A) == 0:
            raise RuntimeError("HSICLasso failed to select features.")
        self.selected_idx_ = np.asarray(self.A, dtype=int)
        return self

    def transform(self, X):
        if self.selected_idx_ is None:
            raise RuntimeError("Call fit before transform.")
        X = np.asarray(X)
        return X[:, self.selected_idx_]


class _FCN(torch.nn.Module):
    def __init__(self, input_dim: int, hidden_dims: List[int]):
        super().__init__()
        layers: List[torch.nn.Module] = []
        prev = input_dim
        for h in hidden_dims:
            layers.append(torch.nn.Linear(prev, h))
            layers.append(torch.nn.ReLU())
            prev = h
        layers.append(torch.nn.Linear(prev, 1))
        self.net = torch.nn.Sequential(*layers)
        self.output_activation = torch.nn.Softplus()

    def forward(self, x):
        return self.output_activation(self.net(x)).squeeze(-1)


class FCNRegressor(BaseEstimator, RegressorMixin):
    """sklearn-compatible FCN regressor optimized with Pearson + Logistic ranking loss."""

    def __init__(
        self,
        network_shape: List[int] | Tuple[int, ...] = (16, 16),
        learning_rate: float = 1e-3,
        early_stopping_rounds: int = 10,
        n_epochs: int = 30,
        coef_pearson: float = 1.0,
        coef_rank: float = 1.0,
        margin: float = 0.0,
        validation_strategy: str = "single_split",
        validation_size: float = 0.2,
        cv_splits: int = 5,
        cv_repeats: int = 1,
        random_state: int = 42,
        verbose: bool = False,
    ):
        self.network_shape = list(network_shape)
        self.learning_rate = learning_rate
        self.early_stopping_rounds = early_stopping_rounds
        self.n_epochs = n_epochs
        self.coef_pearson = coef_pearson
        self.coef_rank = coef_rank
        self.margin = float(margin)
        self.validation_strategy = validation_strategy
        self.validation_size = validation_size
        self.cv_splits = cv_splits
        self.cv_repeats = cv_repeats
        self.random_state = random_state
        self.verbose = verbose
        self.model_: _FCN | None = None
        self.best_epoch_: int | None = None
        self.validation_total_loss_: float | None = None
        self.validation_pearson_: float | None = None
        self.validation_rank_loss_: float | None = None
        self.cv_mean_total_loss_: float | None = None
        self.cv_mean_pearson_: float | None = None
        self.cv_mean_rank_loss_: float | None = None

    def _set_seed(self):
        random.seed(self.random_state)
        np.random.seed(self.random_state)
        torch.manual_seed(self.random_state)

    @staticmethod
    def _pearson_loss(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        pred_c = pred - pred.mean()
        target_c = target - target.mean()
        denom = torch.sqrt(torch.sum(pred_c**2) + eps) * torch.sqrt(torch.sum(target_c**2) + eps)
        r = torch.sum(pred_c * target_c) / denom
        return (1.0 - r) / 2.0

    @staticmethod
    def _logistic_rank_loss(pred: torch.Tensor, stage: torch.Tensor, margin: float = 0.0) -> torch.Tensor:
        # Ranking constraints: stage0 < stage1 and stage1 < stage2 with margin.
        # For each pair, enforce (high - low) >= margin via softplus(margin - (high - low)).
        m = torch.as_tensor(margin, dtype=pred.dtype, device=pred.device)
        losses = []
        for s_low, s_high in [(0, 1), (1, 2)]:
            low = pred[stage == s_low]
            high = pred[stage == s_high]
            if low.numel() == 0 or high.numel() == 0:
                continue
            delta = high[:, None] - low[None, :]
            losses.append(torch.nn.functional.softplus(m - delta).mean())
        if not losses:
            return torch.tensor(0.0, dtype=pred.dtype, device=pred.device)
        return torch.stack(losses).mean()

    def fit(
        self,
        X,
        y,
        stage_labels=None,
    ):
        if float(self.coef_pearson) < 0 or float(self.coef_rank) < 0:
            raise ValueError("coef_pearson and coef_rank must be >= 0.")
        if float(self.coef_pearson) == 0.0 and float(self.coef_rank) == 0.0:
            raise ValueError("coef_pearson and coef_rank cannot both be 0.")

        if stage_labels is None:
            raise ValueError("FCNRegressor.fit requires stage_labels for ranking loss.")
        X_np = np.asarray(X, dtype=np.float32)
        y_np = np.asarray(y, dtype=np.float32).ravel()
        s_np = np.asarray(stage_labels, dtype=np.int64).ravel()
        if X_np.shape[0] != y_np.shape[0] or X_np.shape[0] != s_np.shape[0]:
            raise ValueError("X, y, stage_labels must have same number of rows.")

        self._set_seed()

        n_features = X_np.shape[1]
        if len(self.network_shape) == 0:
            raise ValueError("network_shape cannot be empty.")
        # Accept both [n_features, ...] and hidden-only shape.
        if self.network_shape[0] == n_features:
            hidden_dims = list(self.network_shape[1:])
        else:
            hidden_dims = list(self.network_shape)
        if len(hidden_dims) == 0:
            hidden_dims = [16]

        strategy = str(self.validation_strategy).strip().lower()
        split_pairs: List[Tuple[np.ndarray, np.ndarray]] = []
        if strategy == "single_split":
            if not (0.0 < float(self.validation_size) < 1.0):
                raise ValueError("validation_size must be in (0, 1) when validation_strategy=single_split.")
            splitter = StratifiedShuffleSplit(
                n_splits=1,
                test_size=float(self.validation_size),
                random_state=self.random_state,
            )
            split_pairs = list(splitter.split(X_np, s_np))
        elif strategy == "stratified_cv":
            if int(self.cv_splits) < 2:
                raise ValueError("cv_splits must be >= 2 when validation_strategy=stratified_cv.")
            splitter = RepeatedStratifiedKFold(
                n_splits=int(self.cv_splits),
                n_repeats=max(1, int(self.cv_repeats)),
                random_state=self.random_state,
            )
            split_pairs = list(splitter.split(X_np, s_np))
        else:
            raise ValueError("validation_strategy must be one of {'single_split', 'stratified_cv'}.")

        if len(split_pairs) == 0:
            raise RuntimeError("No validation split generated.")

        split_results: List[Dict[str, Any]] = []
        for tr_idx, val_idx in split_pairs:
            X_tr = torch.from_numpy(X_np[tr_idx])
            y_tr = torch.from_numpy(y_np[tr_idx])
            s_tr = torch.from_numpy(s_np[tr_idx])
            X_val_t = torch.from_numpy(X_np[val_idx])
            y_val_t = torch.from_numpy(y_np[val_idx])
            s_val_t = torch.from_numpy(s_np[val_idx])

            model = _FCN(input_dim=n_features, hidden_dims=hidden_dims)
            optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate)

            best_state = None
            best_total_loss = float("inf")
            best_loss_reg = float("inf")
            best_loss_rank = float("inf")
            best_epoch = 0
            no_improve = 0

            for epoch in range(self.n_epochs):
                model.train()
                optimizer.zero_grad()
                pred_tr = model(X_tr)
                loss_reg_tr = self._pearson_loss(pred_tr, y_tr)
                loss_rank_tr = self._logistic_rank_loss(pred_tr, s_tr, margin=self.margin)
                loss_tr = self.coef_pearson * loss_reg_tr + self.coef_rank * loss_rank_tr
                loss_tr.backward()
                optimizer.step()

                model.eval()
                with torch.no_grad():
                    pred_val = model(X_val_t)
                    loss_reg_val = self._pearson_loss(pred_val, y_val_t)
                    loss_rank_val = self._logistic_rank_loss(pred_val, s_val_t, margin=self.margin)
                    loss_val = float((self.coef_pearson * loss_reg_val + self.coef_rank * loss_rank_val).item())

                if loss_val < best_total_loss - 1e-7:
                    best_total_loss = loss_val
                    best_loss_reg = float(loss_reg_val.item())
                    best_loss_rank = float(loss_rank_val.item())
                    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                    best_epoch = epoch
                    no_improve = 0
                else:
                    no_improve += 1

                if self.verbose:
                    print(
                        f"[FCN] epoch={epoch + 1} "
                        f"train={loss_tr.item():.6f} val={loss_val:.6f}"
                    )
                if no_improve >= self.early_stopping_rounds:
                    break

            if best_state is not None:
                model.load_state_dict(best_state)

            split_results.append(
                {
                    "model": model,
                    "best_epoch": int(best_epoch),
                    "total_loss": float(best_total_loss),
                    "loss_reg": float(best_loss_reg),
                    "loss_rank": float(best_loss_rank),
                    "pearson": float(1.0 - 2.0 * best_loss_reg),
                }
            )

        best_idx = int(np.argmin([r["total_loss"] for r in split_results]))
        best_result = split_results[best_idx]
        self.model_ = best_result["model"]
        self.best_epoch_ = int(best_result["best_epoch"])
        self.validation_total_loss_ = float(best_result["total_loss"])
        self.validation_pearson_ = float(best_result["pearson"])
        self.validation_rank_loss_ = float(best_result["loss_rank"])
        self.cv_mean_total_loss_ = float(np.mean([r["total_loss"] for r in split_results]))
        self.cv_mean_pearson_ = float(np.mean([r["pearson"] for r in split_results]))
        self.cv_mean_rank_loss_ = float(np.mean([r["loss_rank"] for r in split_results]))
        return self

    def predict(self, X):
        if self.model_ is None:
            raise RuntimeError("Call fit before predict.")
        X_t = torch.from_numpy(np.asarray(X, dtype=np.float32))
        self.model_.eval()
        with torch.no_grad():
            pred = self.model_(X_t).cpu().numpy()
        return pred


def pearson_r_np(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-8) -> float:
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    a = y_true - y_true.mean()
    b = y_pred - y_pred.mean()
    denom = np.sqrt(np.sum(a**2) + eps) * np.sqrt(np.sum(b**2) + eps)
    return float(np.sum(a * b) / denom)


def logistic_rank_loss_np(y_pred: np.ndarray, stage: np.ndarray, margin: float = 0.0) -> float:
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    stage = np.asarray(stage, dtype=int).ravel()
    losses = []
    for s_low, s_high in [(0, 1), (1, 2)]:
        low = y_pred[stage == s_low]
        high = y_pred[stage == s_high]
        if low.size == 0 or high.size == 0:
            raise ValueError(f"No samples in stage {s_low} or {s_high}.")
        delta = high[:, None] - low[None, :]
        m = float(margin)
        # softplus(m - delta) = log(1 + exp(m - delta))
        losses.append(np.logaddexp(0.0, m - delta).mean())
    if not losses:
        return 0.0
    return float(np.mean(losses))


def build_pipeline(
    hsiclasso_n_features: int,
    hsiclasso_n_jobs: int,
    network_shape: List[int],
    learning_rate: float,
    early_stopping_rounds: int,
    n_epochs: int,
    coef_pearson: float,
    coef_rank: float,
    margin: float,
    validation_strategy: str,
    validation_size: float,
    cv_splits: int,
    cv_repeats: int,
    random_state: int,
) -> Pipeline:
    feature_selector = HSICLassoTransformer(
        mode="regression",
        num_feat=hsiclasso_n_features,
        n_jobs=hsiclasso_n_jobs,
    )
    fcn_estimator = FCNRegressor(
        network_shape=network_shape,
        learning_rate=learning_rate,
        early_stopping_rounds=early_stopping_rounds,
        n_epochs=n_epochs,
        coef_pearson=coef_pearson,
        coef_rank=coef_rank,
        margin=margin,
        validation_strategy=validation_strategy,
        validation_size=validation_size,
        cv_splits=cv_splits,
        cv_repeats=cv_repeats,
        random_state=random_state,
    )
    return Pipeline(
        [
            ("feature_selector", feature_selector),
            ("fcn_estimator", fcn_estimator),
        ]
    )


def optimize_fold(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    stage_train: pd.Series,
    args,
    fold_dir: str,
) -> Dict[str, Any]:
    n_input_features = X_train.shape[1]
    best_model_path = os.path.join(fold_dir, "best_model.pkl")
    best_so_far_objective = float("inf")
    best_params_so_far: Dict[str, Any] | None = None
    best_inner_pearson_so_far: float = float("nan")
    best_inner_rank_loss_so_far: float = float("nan")

    hsic_low = int(min(HSICLASSO_N_FEATURES_MIN, n_input_features - 1))
    hsic_high = int(min(HSICLASSO_N_FEATURES_MAX, n_input_features - 1))
    if hsic_low > hsic_high:
        raise ValueError(
            "Invalid HSICLasso search space: hsiclasso_n_features_min is larger than the "
            "maximum feasible value under current input dimensionality. "
            f"Got min={hsic_low}, feasible_max={hsic_high}, input_features={n_input_features}."
        )

    def objective(trial: optuna.Trial) -> float:
        nonlocal best_so_far_objective, best_params_so_far, best_inner_pearson_so_far, best_inner_rank_loss_so_far
        hsiclasso_n_features = trial.suggest_int(
            "hsiclasso_n_features",
            hsic_low,
            hsic_high,
            step=10,
        )
        hidden_layers = trial.suggest_int("hidden_layers", 2, 3)
        first_hidden_dim = trial.suggest_categorical("first_hidden_dim", [16, 32])
        last_hidden_dim = trial.suggest_categorical("last_hidden_dim", [2, 4, 8])

        if hidden_layers == 2:
            hidden_dims = [first_hidden_dim, last_hidden_dim]
        else:
            middle_hidden_dim = trial.suggest_categorical("middle_hidden_dim", [2, 4, 8, 16, 32])
            hidden_dims = [first_hidden_dim, middle_hidden_dim, last_hidden_dim]

        learning_rate = trial.suggest_float(
            "learning_rate",
            1e-4,
            3e-3,
            log=True,
        )
        early_stopping_rounds = trial.suggest_int("early_stopping_rounds", 5, 20, step=5)

        validation_strategy = "stratified_cv" if int(args.inner_cv) >= 2 else "single_split"
        pipeline = build_pipeline(
            hsiclasso_n_features=hsiclasso_n_features,
            hsiclasso_n_jobs=args.hsiclasso_n_jobs,
            network_shape=[hsiclasso_n_features, *hidden_dims],
            learning_rate=learning_rate,
            early_stopping_rounds=early_stopping_rounds,
            n_epochs=args.n_epochs,
            coef_pearson=args.coef_pearson,
            coef_rank=args.coef_rank,
            margin=float(args.margin),
            validation_strategy=validation_strategy,
            validation_size=0.2,
            cv_splits=max(2, int(args.inner_cv)),
            cv_repeats=max(1, int(args.inner_repeats)),
            random_state=args.random_state,
        )
        pipeline.fit(
            X_train,
            y_train,
            fcn_estimator__stage_labels=stage_train.values,
        )
        estimator = pipeline.named_steps["fcn_estimator"]
        mean_loss = float(estimator.cv_mean_total_loss_)

        trial.set_user_attr("mean_inner_pearson", float(estimator.cv_mean_pearson_))
        trial.set_user_attr("mean_inner_rank_loss", float(estimator.cv_mean_rank_loss_))
        trial_score = float(mean_loss)
        if trial_score < best_so_far_objective - 1e-12:
            best_so_far_objective = float(trial_score)
            best_params_so_far = dict(trial.params)
            best_inner_pearson_so_far = float(estimator.cv_mean_pearson_)
            best_inner_rank_loss_so_far = float(estimator.cv_mean_rank_loss_)
            joblib.dump(pipeline, best_model_path, compress=3, protocol=4)

        return float(trial_score)

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=args.n_trials, n_jobs=1)
    if not os.path.exists(best_model_path):
        raise RuntimeError("Best model file was not written during optimization.")

    return {
        "params": best_params_so_far,
        "best_inner_total_loss": best_so_far_objective,
        "best_inner_pearson": best_inner_pearson_so_far,
        "best_inner_rank_loss": best_inner_rank_loss_so_far,
        "best_model_path": best_model_path,
    }





