"""
sim_modeller.py
===============
Trains regression / classification models that power the 'ml' simulation mode.

Model families trained per
``(activity, object, object_type, higher_level_activity)`` key:

1. **duration_models**     – regressor predicting duration (median / mean).
2. **duration_std_models** – regressor predicting duration dispersion
                              (fitted on absolute residuals of model 1).
3. **transition_models**   – classifier predicting the *next activity*
                              (including ``'__END__'``).

Available model types (configured via *model_types* list):
    'xgboost'  – XGBRegressor / XGBClassifier
    'linear'   – LinearRegression / LogisticRegression
    'lasso'    – Lasso / LogisticRegression(penalty='l1')
    'mlp'      – MLPRegressor / MLPClassifier

All models in the list are trained for each key, and the **best one on the
temporal test set** is selected automatically.

When *optimize_hyperparams=True* Optuna is used for hyperparameter search
on the **training** portion only.  Otherwise default parameters are used.

A **temporal split** (70 % train / 30 % test, sorted by ``timestamp_start``)
is applied **before** any model fitting.  Final metrics are reported on the
held-out test set.
"""

from __future__ import annotations

import warnings
import numpy as np
import pandas as pd

# ── optional heavy imports ────────────────────────────────────────────────────
try:
    from xgboost import XGBRegressor, XGBClassifier
    _XGBOOST_AVAILABLE = True
except ImportError:
    _XGBOOST_AVAILABLE = False
    warnings.warn(
        "xgboost is not installed – 'xgboost' model type will be skipped. "
        "Install it with:  pip install xgboost",
        ImportWarning, stacklevel=2,
    )

try:
    import optuna
    _OPTUNA_AVAILABLE = True
except ImportError:
    _OPTUNA_AVAILABLE = False

from sklearn.linear_model import LinearRegression, Lasso, LogisticRegression
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    accuracy_score, f1_score,
)

# Minimum number of training samples required before fitting a model.
_MIN_SAMPLES = 5


# ═══════════════════════════════════════════════════════════════════════════════
# Constant baseline "models" — mean & median
# ═══════════════════════════════════════════════════════════════════════════════

class _ConstantPredictor:
    """Sklearn-compatible estimator that always predicts a constant.

    ``strategy='mean'``  → predicts the training mean.
    ``strategy='median'`` → predicts the training median.
    """

    def __init__(self, strategy: str = 'mean'):
        if strategy not in ('mean', 'median'):
            raise ValueError(f"Unknown strategy: {strategy}")
        self.strategy = strategy
        self._value: float = 0.0

    def fit(self, X, y):
        y_arr = np.asarray(y, dtype=float)
        self._value = float(np.mean(y_arr) if self.strategy == 'mean'
                            else np.median(y_arr))
        return self

    def predict(self, X):
        return np.full(len(X), self._value)

    def __repr__(self):
        return f"_ConstantPredictor(strategy='{self.strategy}', value={self._value:.4f})"


# ═══════════════════════════════════════════════════════════════════════════════
# Model factories (default parameters)
# ═══════════════════════════════════════════════════════════════════════════════

def _default_regressor(model_type: str, random_state: int = 42):
    """Return a regressor instance with sensible defaults."""
    if model_type == 'mean':
        return _ConstantPredictor(strategy='mean')
    if model_type == 'median':
        return _ConstantPredictor(strategy='median')
    if model_type == 'xgboost':
        if not _XGBOOST_AVAILABLE:
            raise RuntimeError("xgboost not installed")
        return XGBRegressor(
            n_estimators=100, max_depth=4,
            objective='reg:absoluteerror',
            random_state=random_state, verbosity=0,
        )
    if model_type == 'linear':
        return LinearRegression()
    if model_type == 'lasso':
        return Lasso(alpha=1.0, random_state=random_state, max_iter=5000)
    if model_type == 'mlp':
        return MLPRegressor(
            hidden_layer_sizes=(64, 32), max_iter=500,
            random_state=random_state, early_stopping=True,
            validation_fraction=0.15,
        )
    raise ValueError(f"Unknown model_type: {model_type}")


def _default_classifier(model_type: str, random_state: int = 42):
    """Return a classifier instance with sensible defaults."""
    if model_type in ('mean', 'median'):
        # Baselines aren't meaningful for classification but shouldn't crash
        return _ConstantPredictor(strategy=model_type)
    if model_type == 'xgboost':
        if not _XGBOOST_AVAILABLE:
            raise RuntimeError("xgboost not installed")
        return XGBClassifier(
            n_estimators=100, max_depth=4,
            random_state=random_state, verbosity=0,
            eval_metric='mlogloss', use_label_encoder=False,
        )
    if model_type in ('linear', 'lasso'):
        penalty = 'l1' if model_type == 'lasso' else 'l2'
        return LogisticRegression(
            penalty=penalty, solver='saga', max_iter=5000,
            random_state=random_state, multi_class='multinomial',
        )
    if model_type == 'mlp':
        return MLPClassifier(
            hidden_layer_sizes=(64, 32), max_iter=500,
            random_state=random_state,
        )
    raise ValueError(f"Unknown model_type: {model_type}")


# ═══════════════════════════════════════════════════════════════════════════════
# Optuna objective factories
# ═══════════════════════════════════════════════════════════════════════════════

def _optuna_regressor(trial, model_type: str, random_state: int = 42):
    """Return a regressor with Optuna-suggested hyper-parameters."""
    # Baselines have no hyper-parameters to tune
    if model_type in ('mean', 'median'):
        return _ConstantPredictor(strategy=model_type)
    if model_type == 'xgboost':
        return XGBRegressor(
            n_estimators=trial.suggest_int('n_estimators', 50, 300),
            max_depth=trial.suggest_int('max_depth', 2, 10),
            learning_rate=trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            subsample=trial.suggest_float('subsample', 0.5, 1.0),
            objective='reg:absoluteerror',
            random_state=random_state, verbosity=0,
        )
    if model_type == 'linear':
        return LinearRegression()
    if model_type == 'lasso':
        return Lasso(
            alpha=trial.suggest_float('alpha', 1e-4, 10.0, log=True),
            random_state=random_state, max_iter=5000,
        )
    if model_type == 'mlp':
        n_layers = trial.suggest_int('n_layers', 1, 3)
        layers = tuple(
            trial.suggest_int(f'units_l{i}', 16, 128) for i in range(n_layers)
        )
        return MLPRegressor(
            hidden_layer_sizes=layers,
            learning_rate_init=trial.suggest_float('lr', 1e-4, 1e-2, log=True),
            max_iter=500, random_state=random_state,
            early_stopping=True, validation_fraction=0.15,
        )
    raise ValueError(f"Unknown model_type: {model_type}")


def _optuna_classifier(trial, model_type: str, random_state: int = 42):
    """Return a classifier with Optuna-suggested hyper-parameters."""
    if model_type in ('mean', 'median'):
        return _ConstantPredictor(strategy=model_type)
    if model_type == 'xgboost':
        return XGBClassifier(
            n_estimators=trial.suggest_int('n_estimators', 50, 300),
            max_depth=trial.suggest_int('max_depth', 2, 10),
            learning_rate=trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            subsample=trial.suggest_float('subsample', 0.5, 1.0),
            random_state=random_state, verbosity=0,
            eval_metric='mlogloss', use_label_encoder=False,
        )
    if model_type in ('linear', 'lasso'):
        penalty = 'l1' if model_type == 'lasso' else 'l2'
        return LogisticRegression(
            penalty=penalty, solver='saga', max_iter=5000,
            C=trial.suggest_float('C', 1e-3, 100.0, log=True),
            random_state=random_state, multi_class='multinomial',
        )
    if model_type == 'mlp':
        n_layers = trial.suggest_int('n_layers', 1, 3)
        layers = tuple(
            trial.suggest_int(f'units_l{i}', 16, 128) for i in range(n_layers)
        )
        return MLPClassifier(
            hidden_layer_sizes=layers,
            learning_rate_init=trial.suggest_float('lr', 1e-4, 1e-2, log=True),
            max_iter=500, random_state=random_state,
        )
    raise ValueError(f"Unknown model_type: {model_type}")


# ═══════════════════════════════════════════════════════════════════════════════
# SimModeller
# ═══════════════════════════════════════════════════════════════════════════════

class SimModeller:
    """
    Train and serve ML models for the 'ml' / 'ml_duration_only' simulation
    modes.

    All model types in *model_types* are trained for each activity key,
    and the **best one on the internal validation set** is automatically
    selected.  The winning model is then re-trained on all incoming data.

    Parameters
    ----------
    model_types : list[str]
        List of model types to train and compare.
        Supported: ``'mean'``, ``'median'``, ``'xgboost'``, ``'linear'``,
        ``'lasso'``, ``'mlp'``.
    optimize_hyperparams : bool
        If ``True``, use Optuna to find optimal hyper-parameters on the
        training split.  Requires ``optuna`` to be installed.
        (Ignored for ``'mean'`` and ``'median'`` which have no parameters.)
    n_optuna_trials : int
        Number of Optuna trials per model (only used when
        *optimize_hyperparams* is True).
    val_size : float
        Fraction of data reserved for the internal validation set
        (default 0.20).  Used for model selection only.
    random_state : int
        Random seed for reproducibility.
    """

    def __init__(
        self,
        model_types: list[str] | None = None,
        optimize_hyperparams: bool = False,
        n_optuna_trials: int = 50,
        val_size: float = 0.20,
        train_transitions: bool = True,
        random_state: int = 42,
    ):
        self.model_types = model_types or ['xgboost']
        self.train_transitions = train_transitions
        self.optimize_hyperparams  = optimize_hyperparams
        self.n_optuna_trials       = n_optuna_trials
        self.val_size              = val_size
        self.random_state          = random_state

        # Filter out xgboost if not available
        if not _XGBOOST_AVAILABLE and 'xgboost' in self.model_types:
            print("[SimModeller] xgboost not available – removing from model list.")
            self.model_types = [m for m in self.model_types if m != 'xgboost']

        # ── model stores (best model per key) ─────────────────────────────
        self.duration_models:     dict = {}   # key -> (model, feature_cols, model_type)
        self.duration_std_models: dict = {}   # key -> (model, feature_cols, model_type)
        self.transition_models:   dict = {}   # key -> (model, LabelEncoder, feature_cols, model_type)

        # ── statistical fallbacks (populated from stats_df during train) ──
        self.duration_fallback:   dict = {}   # key -> (dist_name, dist_params)
        self.transition_fallback: dict = {}   # key -> transition dict

        # ── evaluation metrics (internal validation for model selection) ──
        self.duration_val_metrics:   dict = {}   # key -> {mae, rmse, r2, model_type}
        self.transition_val_metrics: dict = {}   # key -> {accuracy, f1, model_type}
        # All model results (for comparison reporting)
        self.duration_all_results:   dict = {}   # key -> {model_type: {mae, rmse, r2}}
        self.transition_all_results: dict = {}   # key -> {model_type: {accuracy, f1}}

        self._trained = False

        # ── label encoder for prev_activity columns ───────────────────────
        self.activity_label_encoder = None   # fitted during train()

        # ── global model (single model across all activities) ─────────────
        self.global_duration_model = None    # (model, feat_cols, cat_encoders, model_type)
        self.global_val_metrics:  dict = {}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _make_key(activity, object_name, object_type, higher_level_activity):
        return (
            str(activity).strip(),
            str(object_name).strip(),
            str(object_type).strip(),
            str(higher_level_activity).strip()
                if pd.notna(higher_level_activity) else None,
        )

    @staticmethod
    def _feature_cols(df: pd.DataFrame) -> list[str]:
        """Return attr_* plus prev_* feature columns."""
        cols = [c for c in df.columns if c.startswith('attr_')]
        for c in ('prev_activity_1', 'prev_duration_1',
                  'prev_activity_2', 'prev_duration_2'):
            if c in df.columns:
                cols.append(c)
        return cols

    def _build_X(self, group: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
        """Return numeric feature matrix (impute missing values with 0).
        Activity-name columns are label-encoded using self.activity_label_encoder."""
        X = group[feature_cols].copy()
        # label-encode prev_activity columns
        for col in ('prev_activity_1', 'prev_activity_2'):
            if col in X.columns and self.activity_label_encoder is not None:
                le = self.activity_label_encoder
                X[col] = X[col].map(
                    lambda v, _le=le: (
                        int(_le.transform([str(v)])[0])
                        if str(v) in _le.classes_ else -1
                    )
                )
        for col in feature_cols:
            X[col] = pd.to_numeric(X[col], errors='coerce')
        return X.fillna(0.0)

    # ------------------------------------------------------------------
    # Internal validation split helper (for model selection only)
    # ------------------------------------------------------------------

    @staticmethod
    def _validation_split(group: pd.DataFrame, val_size: float = 0.20):
        """
        Split *group* into fit / validation (for model selection).
        Sorts by ``timestamp_start`` if available, else uses tail.
        """
        if 'timestamp_start' in group.columns:
            sorted_g = group.sort_values('timestamp_start').reset_index(drop=True)
        else:
            sorted_g = group.reset_index(drop=True)
        split_idx = int(len(sorted_g) * (1 - val_size))
        split_idx = max(1, min(split_idx, len(sorted_g) - 1))
        return sorted_g.iloc[:split_idx], sorted_g.iloc[split_idx:]

    # ------------------------------------------------------------------
    # Single-model train helpers
    # ------------------------------------------------------------------

    def _fit_regressor(self, model_type, X_train, y_train):
        """Train one regressor (with or without Optuna) and return it."""
        if self.optimize_hyperparams and _OPTUNA_AVAILABLE:
            return self._optuna_fit_regressor(model_type, X_train, y_train)
        model = _default_regressor(model_type, self.random_state)
        model.fit(X_train, y_train)
        return model

    def _fit_classifier(self, model_type, X_train, y_enc_train):
        """Train one classifier (with or without Optuna) and return it."""
        if self.optimize_hyperparams and _OPTUNA_AVAILABLE:
            return self._optuna_fit_classifier(model_type, X_train, y_enc_train)
        model = _default_classifier(model_type, self.random_state)
        model.fit(X_train, y_enc_train)
        return model

    # ------------------------------------------------------------------
    # Optuna fitting helpers
    # ------------------------------------------------------------------

    def _optuna_fit_regressor(self, model_type, X_train, y_train):
        """Run Optuna study for one model type and return the best regressor
        re-fitted on full training data."""
        optuna.logging.set_verbosity(optuna.logging.WARNING)

        def objective(trial):
            model = _optuna_regressor(trial, model_type, self.random_state)
            n = len(X_train)
            split = int(n * 0.8)
            X_tr, X_val = X_train.iloc[:split], X_train.iloc[split:]
            y_tr, y_val = y_train.iloc[:split], y_train.iloc[split:]
            model.fit(X_tr, y_tr)
            return -mean_absolute_error(y_val, model.predict(X_val))

        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=self.n_optuna_trials,
                       show_progress_bar=False)

        best_model = _optuna_regressor(
            study.best_trial, model_type, self.random_state
        )
        best_model.fit(X_train, y_train)
        return best_model

    def _optuna_fit_classifier(self, model_type, X_train, y_enc_train):
        """Run Optuna study for one model type and return the best classifier
        re-fitted on full training data."""
        optuna.logging.set_verbosity(optuna.logging.WARNING)

        def objective(trial):
            model = _optuna_classifier(trial, model_type, self.random_state)
            n = len(X_train)
            split = int(n * 0.8)
            X_tr, X_val = X_train.iloc[:split], X_train.iloc[split:]
            y_tr, y_val = y_enc_train[:split], y_enc_train[split:]
            model.fit(X_tr, y_tr)
            return accuracy_score(y_val, model.predict(X_val))

        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=self.n_optuna_trials,
                       show_progress_bar=False)

        best_model = _optuna_classifier(
            study.best_trial, model_type, self.random_state
        )
        best_model.fit(X_train, y_enc_train)
        return best_model

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def train(self, raw_df: pd.DataFrame, stats_df: pd.DataFrame) -> None:
        """
        Train all ML models.

        The incoming ``raw_df`` is assumed to be the **train portion** only
        (the pipeline handles the real train/test split).  Internally, a
        small validation split (``val_size``) is used only for model
        selection among ``model_types``.  The winning model is then
        **re-trained on all incoming data** before being stored.

        Parameters
        ----------
        raw_df : pd.DataFrame
            Output from ``sim_extractor.extract_process`` – one row per
            activity instance with columns ``activity``, ``object``,
            ``object_type``, ``higher_level_activity``, ``duration``,
            ``next_activity``, ``timestamp_start``, and ``attr_*``.
        stats_df : pd.DataFrame
            The statistical summary from ``sim_extractor.extract_process``.
            Used to populate fallback parameters.
        """
        # ── populate statistical fallbacks ────────────────────────────────
        for _, row in stats_df.iterrows():
            key = self._make_key(
                row['activity'], row['object'],
                row['object_type'], row['higher_level_activity']
            )
            self.duration_fallback[key] = (
                row.get('dist_name', 'norm'),
                row.get('dist_params', (row['duration'], row['duration_std'])),
            )
            self.transition_fallback[key] = row['transition']

        if self.optimize_hyperparams and not _OPTUNA_AVAILABLE:
            print("[SimModeller] WARNING: optuna not installed – falling back "
                  "to default hyper-parameters.")
            self.optimize_hyperparams = False

        if not self.model_types:
            print("[SimModeller] No model types available – using statistical "
                  "fallbacks only.")
            self._trained = True
            return

        # ── fit label encoder for prev_activity columns ───────────────────
        all_activities = set()
        for col in ('prev_activity_1', 'prev_activity_2', 'activity'):
            if col in raw_df.columns:
                all_activities |= set(raw_df[col].dropna().astype(str).unique())
        all_activities.add('__NONE__')
        self.activity_label_encoder = LabelEncoder()
        self.activity_label_encoder.fit(sorted(all_activities))

        feature_cols = self._feature_cols(raw_df)
        if not feature_cols:
            print("[SimModeller] No feature columns found in raw_df – "
                  "using statistical fallbacks only.")
            self._trained = True
            return

        group_keys = ['activity', 'object', 'object_type',
                      'higher_level_activity']
        grouped = raw_df.groupby(group_keys, dropna=False)

        dur_trained = 0
        tr_trained  = 0

        print(f"\n[SimModeller] Model types      : {self.model_types}")
        print(f"[SimModeller] Optuna tuning    : {self.optimize_hyperparams}")
        print(f"[SimModeller] Internal val %   : {self.val_size:.0%}")
        print(f"[SimModeller] Feature columns  : {feature_cols}")

        for key_vals, group in grouped:
            key = tuple(
                str(k).strip() if pd.notna(k) else None
                for k in key_vals
            )

            if len(group) < _MIN_SAMPLES:
                continue

            # ── internal validation split (for model selection only) ──────
            fit_g, val_g = self._validation_split(group, self.val_size)

            X_fit = self._build_X(fit_g, feature_cols)
            X_val = self._build_X(val_g, feature_cols)
            # Full data (for re-training the winning model)
            X_all = self._build_X(group, feature_cols)

            # skip if all features are constant (no signal)
            if (X_fit.nunique() <= 1).all():
                continue

            print(f"\n  Key: {key}")
            print(f"    Fit size: {len(fit_g)}, Val size: {len(val_g)}, Total: {len(group)}")

            # ==============================================================
            # 1. Duration model — train ALL types, pick best by val MAE
            # ==============================================================
            y_fit_dur = fit_g['duration'].astype(float)
            y_val_dur = val_g['duration'].astype(float)
            y_all_dur = group['duration'].astype(float)

            best_dur_mae   = float('inf')
            best_dur_type  = None
            best_dur_metrics = None
            dur_results = {}

            for mtype in self.model_types:
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        model = self._fit_regressor(mtype, X_fit, y_fit_dur)

                    # evaluate on validation
                    if len(X_val) > 0:
                        y_pred = model.predict(X_val)
                        mae  = mean_absolute_error(y_val_dur, y_pred)
                        rmse = np.sqrt(mean_squared_error(y_val_dur, y_pred))
                        r2   = r2_score(y_val_dur, y_pred)
                        dur_results[mtype] = {
                            'mae': mae, 'rmse': rmse, 'r2': r2,
                        }
                        print(f"    Duration [{mtype:>8s}]  "
                              f"MAE={mae:.3f}  RMSE={rmse:.3f}  R²={r2:.3f}")

                        if mae < best_dur_mae:
                            best_dur_mae  = mae
                            best_dur_type = mtype
                            best_dur_metrics = dur_results[mtype]
                    else:
                        if best_dur_type is None:
                            best_dur_type = mtype

                except Exception as exc:
                    print(f"    Duration [{mtype:>8s}]  FAILED: {exc}")

            # Re-train the winning model on ALL incoming data
            if best_dur_type is not None:
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        final_dur_model = self._fit_regressor(
                            best_dur_type, X_all, y_all_dur
                        )
                    self.duration_models[key] = (final_dur_model, feature_cols,
                                                 best_dur_type)
                    dur_trained += 1
                    if best_dur_metrics:
                        self.duration_val_metrics[key] = {
                            **best_dur_metrics, 'model_type': best_dur_type,
                        }
                    if dur_results:
                        self.duration_all_results[key] = dur_results

                    print(f"    ✓ Best duration model: {best_dur_type}"
                          + (f"  (val MAE={best_dur_mae:.3f})"
                             if best_dur_mae < float('inf') else ""))

                    # ── Duration std model (on abs residuals) — same type ─
                    residuals = np.abs(
                        y_all_dur.values - final_dur_model.predict(X_all)
                    )
                    std_model = self._fit_regressor(
                        best_dur_type, X_all, pd.Series(residuals)
                    )
                    self.duration_std_models[key] = (std_model,
                                                     feature_cols,
                                                     best_dur_type)
                except Exception as exc:
                    print(f"    [!] Final duration re-train failed: {exc}")

            # ==============================================================
            # 2. Transition model — train ALL types, pick best by accuracy
            #    (skipped when train_transitions=False, e.g. ml_duration_only)
            # ==============================================================
            if not self.train_transitions:
                continue

            y_fit_tr = fit_g['next_activity'].astype(str)
            y_val_tr = val_g['next_activity'].astype(str)
            y_all_tr = group['next_activity'].astype(str)

            if y_all_tr.nunique() < 2:
                continue

            le = LabelEncoder()
            le.fit(y_all_tr)  # fit on ALL classes so nothing is unknown later
            y_enc_fit = le.transform(y_fit_tr)

            best_tr_acc  = -1.0
            best_tr_type = None
            best_tr_metrics = None
            tr_results = {}

            for mtype in self.model_types:
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        model = self._fit_classifier(
                            mtype, X_fit, y_enc_fit
                        )

                    # evaluate on validation
                    if len(X_val) > 0:
                        known_mask = y_val_tr.isin(le.classes_)
                        if known_mask.sum() > 0:
                            y_enc_val = le.transform(y_val_tr[known_mask])
                            y_pred_tr = model.predict(X_val[known_mask])
                            acc = accuracy_score(y_enc_val, y_pred_tr)
                            f1  = f1_score(y_enc_val, y_pred_tr,
                                           average='weighted',
                                           zero_division=0)
                            tr_results[mtype] = {
                                'accuracy': acc, 'f1': f1,
                            }
                            print(f"    Transition [{mtype:>8s}]  "
                                  f"Acc={acc:.3f}  F1={f1:.3f}")

                            if acc > best_tr_acc:
                                best_tr_acc  = acc
                                best_tr_type = mtype
                                best_tr_metrics = tr_results[mtype]
                        else:
                            if best_tr_type is None:
                                best_tr_type = mtype
                    else:
                        if best_tr_type is None:
                            best_tr_type = mtype

                except Exception as exc:
                    print(f"    Transition [{mtype:>8s}]  FAILED: {exc}")

            # Re-train the winning classifier on ALL incoming data
            if best_tr_type is not None:
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        y_enc_all = le.transform(y_all_tr)
                        final_tr_model = self._fit_classifier(
                            best_tr_type, X_all, y_enc_all
                        )
                    self.transition_models[key] = (final_tr_model, le,
                                                   feature_cols, best_tr_type)
                    tr_trained += 1
                    if best_tr_metrics:
                        self.transition_val_metrics[key] = {
                            **best_tr_metrics, 'model_type': best_tr_type,
                        }
                    if tr_results:
                        self.transition_all_results[key] = tr_results

                    print(f"    ✓ Best transition model: {best_tr_type}"
                          + (f"  (val Acc={best_tr_acc:.3f})"
                             if best_tr_acc >= 0 else ""))
                except Exception as exc:
                    print(f"    [!] Final transition re-train failed: {exc}")

        print(
            f"\n[SimModeller] Training complete – "
            f"{dur_trained} duration models, {tr_trained} transition models."
        )

        # ── Global model (single model across all activities) ─────────────
        self._train_global_duration_model(raw_df)

        self._trained = True

    # ------------------------------------------------------------------
    # Global model training
    # ------------------------------------------------------------------

    def _train_global_duration_model(self, raw_df: pd.DataFrame):
        """Train a single duration model on ALL rows (all activity keys).

        Features include label-encoded activity/object/type/higher-level
        plus attr_* and prev_* columns.  This enables cross-activity
        learning that per-key models cannot do.
        """
        import warnings
        print("\n  ── Training Global Duration Model ──")

        df = raw_df.copy()
        cat_cols = ['activity', 'object', 'object_type', 'higher_level_activity']
        cat_encoders: dict = {}
        global_feat_cols: list[str] = []

        # ── encode categorical columns ────────────────────────────────────
        for col in cat_cols:
            feat = f'feat_{col}'
            if col in df.columns:
                le = LabelEncoder()
                vals = df[col].fillna('__NONE__').astype(str)
                df[feat] = le.fit_transform(vals)
                cat_encoders[col] = le
                global_feat_cols.append(feat)

        # ── prev_activity columns (use activity_label_encoder) ────────────
        for col in ('prev_activity_1', 'prev_activity_2'):
            if col in df.columns and self.activity_label_encoder is not None:
                le = self.activity_label_encoder
                df[col] = df[col].map(
                    lambda v, _le=le: (
                        int(_le.transform([str(v)])[0])
                        if str(v) in _le.classes_ else -1
                    )
                )
                global_feat_cols.append(col)

        # ── prev_duration columns ─────────────────────────────────────────
        for col in ('prev_duration_1', 'prev_duration_2'):
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
                global_feat_cols.append(col)

        # ── attr_* columns ────────────────────────────────────────────────
        for col in df.columns:
            if col.startswith('attr_'):
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
                global_feat_cols.append(col)

        if not global_feat_cols:
            print("    No features for global model — skipping.")
            return

        X_all = df[global_feat_cols].astype(float).fillna(0)
        y_all = df['duration'].astype(float)

        print(f"    Features ({len(global_feat_cols)}): {global_feat_cols}")
        print(f"    Training samples: {len(X_all)}")

        # ── validation split ──────────────────────────────────────────────
        fit_g, val_g = self._validation_split(df, self.val_size)
        X_fit = fit_g[global_feat_cols].astype(float).fillna(0)
        y_fit = fit_g['duration'].astype(float)
        X_val = val_g[global_feat_cols].astype(float).fillna(0)
        y_val = val_g['duration'].astype(float)

        best_mae  = float('inf')
        best_type = None
        best_metrics = None

        for mtype in self.model_types:
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    model = self._fit_regressor(mtype, X_fit, y_fit)

                if len(X_val) > 0:
                    y_pred = model.predict(X_val)
                    mae  = mean_absolute_error(y_val, y_pred)
                    rmse = np.sqrt(mean_squared_error(y_val, y_pred))
                    r2   = r2_score(y_val, y_pred)
                    print(f"    Global [{mtype:>8s}]  "
                          f"MAE={mae:.3f}  RMSE={rmse:.3f}  R²={r2:.3f}")
                    if mae < best_mae:
                        best_mae    = mae
                        best_type   = mtype
                        best_metrics = {'mae': mae, 'rmse': rmse, 'r2': r2}
                else:
                    if best_type is None:
                        best_type = mtype
            except Exception as exc:
                print(f"    Global [{mtype:>8s}]  FAILED: {exc}")

        # ── re-train winner on all data ───────────────────────────────────
        if best_type is not None:
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    final_model = self._fit_regressor(
                        best_type, X_all.fillna(0), y_all,
                    )
                self.global_duration_model = (
                    final_model, global_feat_cols, cat_encoders, best_type,
                )
                self.global_val_metrics = best_metrics or {}
                print(f"    ✓ Best global model: {best_type}"
                      + (f"  (val MAE={best_mae:.3f})"
                         if best_mae < float('inf') else ""))
            except Exception as exc:
                print(f"    [!] Global model re-train failed: {exc}")
        else:
            print("    No global model could be trained.")

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict_duration(
        self,
        activity: str,
        object_name: str,
        object_type: str,
        higher_level_activity,
        object_attributes: dict,
    ) -> float | None:
        """
        Predict a sampled duration (minutes) for one activity instance.
        The prediction is centred on the model's point estimate, with noise
        added from the duration-std model.

        Returns ``None`` when no ML model is available for the given key,
        signalling the caller to fall back to statistical mode.
        """
        key = self._make_key(activity, object_name, object_type,
                             higher_level_activity)

        if key not in self.duration_models:
            return None

        dur_model, feature_cols, _mtype = self.duration_models[key]
        features = self._attrs_to_features(object_attributes, feature_cols)
        X = pd.DataFrame([features])[feature_cols]
        for col in X.columns:
            X[col] = pd.to_numeric(X[col], errors='coerce')
        X = X.fillna(0.0)

        median_pred = float(dur_model.predict(X)[0])

        std_pred = 0.0
        if key in self.duration_std_models:
            std_model, _, _ = self.duration_std_models[key]
            std_pred = max(0.0, float(std_model.predict(X)[0]))

        if std_pred > 0:
            sampled = np.random.normal(median_pred, std_pred)
        else:
            sampled = median_pred

        return max(0.1, float(sampled))

    def predict_duration_median(
        self,
        activity: str,
        object_name: str,
        object_type: str,
        higher_level_activity,
        object_attributes: dict,
        activity_history: list | None = None,
    ) -> float | None:
        """
        Return the raw ML duration prediction **without** adding
        ML-predicted noise (std).

        Used by ``'ml_duration_only'`` and
        ``'ml_duration_only_with_activity_past'`` modes.

        Parameters
        ----------
        activity_history : list[tuple[str, float]] | None
            The last N activities as ``[(activity_name, duration_min), ...]``
            ordered most-recent-first.  When supplied the ``prev_activity_*``
            and ``prev_duration_*`` features are populated from this list.

        Returns ``None`` when no ML model is available for the given key.
        """
        key = self._make_key(activity, object_name, object_type,
                             higher_level_activity)

        if key not in self.duration_models:
            return None

        dur_model, feature_cols, _mtype = self.duration_models[key]
        features = self._attrs_to_features(object_attributes, feature_cols)

        # ── fill lag features from activity_history ────────────────────────
        if activity_history is not None:
            le = self.activity_label_encoder
            for i in range(2):
                act_col = f'prev_activity_{i+1}'
                dur_col = f'prev_duration_{i+1}'
                if act_col in feature_cols:
                    if i < len(activity_history):
                        act_name, act_dur = activity_history[i]
                        encoded = -1
                        if le is not None and str(act_name) in le.classes_:
                            encoded = int(le.transform([str(act_name)])[0])
                        features[act_col] = encoded
                        features[dur_col] = act_dur
                    else:
                        features[act_col] = -1
                        features[dur_col] = 0.0

        X = pd.DataFrame([features])[feature_cols]
        for col in X.columns:
            X[col] = pd.to_numeric(X[col], errors='coerce')
        X = X.fillna(0.0)

        median_pred = float(dur_model.predict(X)[0])
        return max(0.1, median_pred)

    def predict_duration_global(
        self,
        activity: str,
        object_name: str,
        object_type: str,
        higher_level_activity,
        object_attributes: dict,
        activity_history: list | None = None,
    ) -> float | None:
        """
        Predict duration using the single global model (one model for
        all activities).  Returns ``None`` if no global model was trained.
        """
        if self.global_duration_model is None:
            return None

        model, global_feat_cols, cat_encoders, _mtype = self.global_duration_model
        features: dict = {}

        # ── encode categorical features ───────────────────────────────────
        cat_values = {
            'activity':              str(activity).strip(),
            'object':                str(object_name).strip(),
            'object_type':           str(object_type).strip(),
            'higher_level_activity': (str(higher_level_activity).strip()
                                      if pd.notna(higher_level_activity)
                                      else '__NONE__'),
        }
        for col, val in cat_values.items():
            feat = f'feat_{col}'
            if feat in global_feat_cols and col in cat_encoders:
                le = cat_encoders[col]
                if val in le.classes_:
                    features[feat] = int(le.transform([val])[0])
                else:
                    features[feat] = -1

        # ── prev_activity / prev_duration features ────────────────────────
        if activity_history is not None:
            le = self.activity_label_encoder
            for i in range(2):
                act_col = f'prev_activity_{i+1}'
                dur_col = f'prev_duration_{i+1}'
                if act_col in global_feat_cols:
                    if i < len(activity_history):
                        act_name, act_dur = activity_history[i]
                        encoded = -1
                        if le is not None and str(act_name) in le.classes_:
                            encoded = int(le.transform([str(act_name)])[0])
                        features[act_col] = encoded
                        features[dur_col] = act_dur
                    else:
                        features[act_col] = -1
                        features[dur_col] = 0.0

        # ── attr_* features ───────────────────────────────────────────────
        for col in global_feat_cols:
            if col.startswith('attr_'):
                features[col] = object_attributes.get(col[5:], 0)

        # ── fill any missing columns with 0 ───────────────────────────────
        for col in global_feat_cols:
            if col not in features:
                features[col] = 0

        X = pd.DataFrame([features])[global_feat_cols]
        for col in X.columns:
            X[col] = pd.to_numeric(X[col], errors='coerce')
        X = X.fillna(0.0)

        pred = float(model.predict(X)[0])
        return max(0.1, pred)

    def predict_transitions(
        self,
        activity: str,
        object_name: str,
        object_type: str,
        higher_level_activity,
        object_attributes: dict,
    ) -> dict | None:
        """
        Predict transition probabilities as ``{next_activity: probability}``.

        Returns ``None`` when no ML model is available.
        """
        key = self._make_key(activity, object_name, object_type,
                             higher_level_activity)

        if key not in self.transition_models:
            return None

        tr_model, le, feature_cols, _mtype = self.transition_models[key]
        features = self._attrs_to_features(object_attributes, feature_cols)
        X = pd.DataFrame([features])[feature_cols]
        for col in X.columns:
            X[col] = pd.to_numeric(X[col], errors='coerce')
        X = X.fillna(0.0)

        probs = tr_model.predict_proba(X)[0]
        return {
            str(cls): float(prob)
            for cls, prob in zip(le.classes_, probs)
        }

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _attrs_to_features(object_attributes: dict,
                           feature_cols: list[str]) -> dict:
        """Map ``object_attributes`` dict to the ``attr_*`` / ``prev_*``
        feature space.  ``prev_*`` columns default to 0 here and are
        overridden by the caller when activity history is available."""
        feats = {}
        for col in feature_cols:
            if col.startswith('attr_'):
                feats[col] = object_attributes.get(col[5:], 0)
            else:
                feats[col] = 0   # prev_* defaults; overridden by caller
        return feats

    def summary(self) -> str:
        lines = [
            f"SimModeller summary  (model_types={self.model_types}, "
            f"optuna={self.optimize_hyperparams})",
            f"  Duration models  : {len(self.duration_models)}",
            f"  Duration std mdls: {len(self.duration_std_models)}",
            f"  Transition models: {len(self.transition_models)}",
            f"  Fallback entries : {len(self.duration_fallback)} (duration),"
            f" {len(self.transition_fallback)} (transition)",
        ]

        if self.duration_val_metrics:
            lines.append("\n  ── Best duration model per key (on val set) ──")
            for key, m in self.duration_val_metrics.items():
                lines.append(
                    f"    {key}  →  [{m['model_type']}]  "
                    f"MAE={m['mae']:.3f}  RMSE={m['rmse']:.3f}  "
                    f"R²={m['r2']:.3f}"
                )

        if self.duration_all_results:
            lines.append("\n  ── All duration model comparisons ──")
            for key, results in self.duration_all_results.items():
                lines.append(f"    {key}:")
                for mtype, m in results.items():
                    best_marker = " ★" if (
                        key in self.duration_val_metrics
                        and self.duration_val_metrics[key]['model_type'] == mtype
                    ) else ""
                    lines.append(
                        f"      {mtype:>8s}:  MAE={m['mae']:.3f}  "
                        f"RMSE={m['rmse']:.3f}  R²={m['r2']:.3f}{best_marker}"
                    )

        if self.transition_val_metrics:
            lines.append("\n  ── Best transition model per key (on val set) ──")
            for key, m in self.transition_val_metrics.items():
                lines.append(
                    f"    {key}  →  [{m['model_type']}]  "
                    f"Acc={m['accuracy']:.3f}  F1={m['f1']:.3f}"
                )

        if self.transition_all_results:
            lines.append("\n  ── All transition model comparisons ──")
            for key, results in self.transition_all_results.items():
                lines.append(f"    {key}:")
                for mtype, m in results.items():
                    best_marker = " ★" if (
                        key in self.transition_val_metrics
                        and self.transition_val_metrics[key]['model_type'] == mtype
                    ) else ""
                    lines.append(
                        f"      {mtype:>8s}:  Acc={m['accuracy']:.3f}  "
                        f"F1={m['f1']:.3f}{best_marker}"
                    )

        return "\n".join(lines)
