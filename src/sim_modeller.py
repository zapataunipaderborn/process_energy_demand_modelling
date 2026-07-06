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


class _ConstantClassifier:
    """Sklearn-compatible classifier baseline that always predicts the most
    frequent (mode) class seen during fit — the only sensible constant
    baseline for classification. Used for the 'mean'/'median' model_type
    entries when training a classifier (arithmetic mean/median of
    label-encoded class integers is meaningless and can even produce a
    non-integer, non-class prediction)."""

    def __init__(self, strategy: str = 'mean'):
        self.strategy = strategy  # kept only for repr/consistency; unused
        self._value = 0

    def fit(self, X, y):
        y_arr = np.asarray(y)
        values, counts = np.unique(y_arr, return_counts=True)
        self._value = values[np.argmax(counts)]
        return self

    def predict(self, X):
        return np.full(len(X), self._value)

    # Deliberately no predict_proba: callers (predict_transitions /
    # predict_transitions_wip_aware) check hasattr(model, 'predict_proba')
    # and fall back to the statistical distribution when absent — the right
    # behaviour here, since a single constant class isn't a real distribution.

    def __repr__(self):
        return f"_ConstantClassifier(value={self._value!r})"


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
            n_jobs=1,
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
        # Constant-majority-class baseline (predicting the arithmetic
        # mean/median of label-encoded class integers is meaningless for
        # classification — both map to the same majority-class strategy here).
        return _ConstantClassifier(strategy=model_type)
    if model_type == 'xgboost':
        if not _XGBOOST_AVAILABLE:
            raise RuntimeError("xgboost not installed")
        return XGBClassifier(
            n_estimators=100, max_depth=4,
            random_state=random_state, verbosity=0,
            eval_metric='mlogloss', use_label_encoder=False,
            n_jobs=1,
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
            n_jobs=1,
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
        # No hyperparameters to tune — same majority-class baseline as above.
        return _ConstantClassifier(strategy=model_type)
    if model_type == 'xgboost':
        return XGBClassifier(
            n_estimators=trial.suggest_int('n_estimators', 50, 300),
            max_depth=trial.suggest_int('max_depth', 2, 10),
            learning_rate=trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            subsample=trial.suggest_float('subsample', 0.5, 1.0),
            random_state=random_state, verbosity=0,
            eval_metric='mlogloss', use_label_encoder=False,
            n_jobs=1,
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
        train_waiting: bool = True,
        random_state: int = 42,
    ):
        self.model_types = model_types or ['xgboost']
        self.train_transitions = train_transitions
        self.train_waiting = train_waiting
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
        self.waiting_models:      dict = {}   # key -> (model, feature_cols, model_type)
        self.waiting_std_models:  dict = {}   # key -> (model, feature_cols, model_type)
        # WIP/RO-aware transition classifiers — populated by train_wip_transitions().
        # Only kept per-key when they beat the statistical baseline on validation
        # (see train_wip_transitions); otherwise absent, so
        # predict_transitions_wip_aware() returns None and the caller falls
        # back to the frequency-based statistical transitions.
        self.transition_wip_models: dict = {}   # key -> (model, LabelEncoder, feature_cols, model_type)

        # ── statistical fallbacks (populated from stats_df during train) ──
        self.duration_fallback:   dict = {}   # key -> (dist_name, dist_params)
        self.transition_fallback: dict = {}   # key -> transition dict

        # ── evaluation metrics (internal validation for model selection) ──
        self.duration_val_metrics:   dict = {}   # key -> {mae, rmse, r2, model_type}
        self.transition_val_metrics: dict = {}   # key -> {accuracy, f1, model_type}
        self.waiting_val_metrics:    dict = {}   # key -> {mae, rmse, r2, model_type}
        # All model results (for comparison reporting)
        self.duration_all_results:   dict = {}   # key -> {model_type: {mae, rmse, r2}}
        self.transition_all_results: dict = {}   # key -> {model_type: {accuracy, f1}}
        self.waiting_all_results:    dict = {}   # key -> {model_type: {mae, rmse, r2}}

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
        """Return attr_* plus prev_* plus wip/ro feature columns."""
        cols = [c for c in df.columns if c.startswith('attr_')]
        for c in ('prev_activity_1', 'prev_duration_1',
                  'prev_activity_2', 'prev_duration_2',
                  'wip', 'ro'):
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

        sampler = optuna.samplers.TPESampler(seed=self.random_state)
        study = optuna.create_study(direction='maximize', sampler=sampler)
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

            # XGBoost requires the labels passed to fit() to be exactly
            # 0..k-1 for whatever classes appear in THIS call. A naive
            # positional split can leave y_tr with a non-contiguous subset
            # of the globally-encoded classes (e.g. {0,2,5,6}), which
            # XGBClassifier rejects outright. Re-encode locally to a
            # compact, contiguous range for this split.
            local_classes = np.unique(y_tr)
            if len(local_classes) < 2:
                return 0.0  # nothing to learn from a single-class split
            local_map = {c: i for i, c in enumerate(local_classes)}
            y_tr_local = np.array([local_map[c] for c in y_tr])

            # Validation rows whose true class never appeared in y_tr are
            # unlearnable for this model — exclude them rather than crash.
            val_mask = np.isin(y_val, local_classes)
            if val_mask.sum() == 0:
                return 0.0
            y_val_local = np.array([local_map[c] for c in np.asarray(y_val)[val_mask]])
            X_val_known = X_val.iloc[val_mask] if hasattr(X_val, 'iloc') else X_val[val_mask]

            model.fit(X_tr, y_tr_local)
            return accuracy_score(y_val_local, model.predict(X_val_known))

        sampler = optuna.samplers.TPESampler(seed=self.random_state)
        study = optuna.create_study(direction='maximize', sampler=sampler)
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

        dur_trained  = 0
        tr_trained   = 0
        wait_trained = 0

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
            # 1b. Waiting-time model — same shape as the duration model,
            #     but predicts the queueing/contention gap *before* this
            #     activity starts (skipped if no 'waiting_time' column or
            #     train_waiting=False).
            # ==============================================================
            if self.train_waiting and 'waiting_time' in group.columns:
                y_fit_wait = fit_g['waiting_time'].astype(float)
                y_val_wait = val_g['waiting_time'].astype(float)
                y_all_wait = group['waiting_time'].astype(float)

                best_wait_mae    = float('inf')
                best_wait_type   = None
                best_wait_metrics = None
                wait_results = {}

                for mtype in self.model_types:
                    try:
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore")
                            model = self._fit_regressor(mtype, X_fit, y_fit_wait)

                        if len(X_val) > 0:
                            y_pred = model.predict(X_val)
                            mae  = mean_absolute_error(y_val_wait, y_pred)
                            rmse = np.sqrt(mean_squared_error(y_val_wait, y_pred))
                            r2   = r2_score(y_val_wait, y_pred)
                            wait_results[mtype] = {'mae': mae, 'rmse': rmse, 'r2': r2}
                            print(f"    Waiting   [{mtype:>8s}]  "
                                  f"MAE={mae:.3f}  RMSE={rmse:.3f}  R²={r2:.3f}")

                            if mae < best_wait_mae:
                                best_wait_mae   = mae
                                best_wait_type  = mtype
                                best_wait_metrics = wait_results[mtype]
                        else:
                            if best_wait_type is None:
                                best_wait_type = mtype
                    except Exception as exc:
                        print(f"    Waiting   [{mtype:>8s}]  FAILED: {exc}")

                if best_wait_type is not None:
                    try:
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore")
                            final_wait_model = self._fit_regressor(
                                best_wait_type, X_all, y_all_wait
                            )
                        self.waiting_models[key] = (final_wait_model, feature_cols,
                                                    best_wait_type)
                        wait_trained += 1
                        if best_wait_metrics:
                            self.waiting_val_metrics[key] = {
                                **best_wait_metrics, 'model_type': best_wait_type,
                            }
                        if wait_results:
                            self.waiting_all_results[key] = wait_results

                        print(f"    ✓ Best waiting model: {best_wait_type}"
                              + (f"  (val MAE={best_wait_mae:.3f})"
                                 if best_wait_mae < float('inf') else ""))

                        residuals = np.abs(
                            y_all_wait.values - final_wait_model.predict(X_all)
                        )
                        wait_std_model = self._fit_regressor(
                            best_wait_type, X_all, pd.Series(residuals)
                        )
                        self.waiting_std_models[key] = (wait_std_model,
                                                        feature_cols,
                                                        best_wait_type)
                    except Exception as exc:
                        print(f"    [!] Final waiting re-train failed: {exc}")

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
            f"{dur_trained} duration models, {tr_trained} transition models, "
            f"{wait_trained} waiting-time models."
        )

        # ── Global model (single model across all activities) ─────────────
        self._train_global_duration_model(raw_df)

        self._trained = True

    # ------------------------------------------------------------------
    # WIP/RO-aware transition classifiers (for petri_net_wip_branching_aware)
    # ------------------------------------------------------------------

    def train_wip_transitions(self, raw_df: pd.DataFrame, stats_df: pd.DataFrame) -> None:
        """
        Train WIP/RO-aware transition classifiers, one per
        (activity, object, object_type, higher_level_activity) key, predicting
        the next activity (including '__END__').

        A model is kept **only if it beats the frequency-based statistical
        baseline's accuracy** on the same internal validation split. Groups
        with fewer than ``_MIN_SAMPLES`` rows, or where no ``model_types``
        candidate beats the baseline, are left unset — at prediction time
        ``predict_transitions_wip_aware`` then returns None for that key,
        signalling the caller to fall back to the statistical distribution
        (``self.transition_fallback`` / ``activity_config[...]['transitions']``).

        This is additive and independent of ``train()``/``transition_models``
        (used by the plain 'ml' mode) — calling this does not change the
        behaviour of any other simulation mode.
        """
        if not self.model_types:
            print("[SimModeller] No model types available — "
                  "skipping WIP-transition training.")
            return

        feature_cols = self._feature_cols(raw_df)
        if not feature_cols:
            print("[SimModeller] No feature columns found — "
                  "skipping WIP-transition training.")
            return

        # Populate the statistical fallback in case train() hasn't run yet.
        for _, row in stats_df.iterrows():
            key = self._make_key(
                row['activity'], row['object'],
                row['object_type'], row['higher_level_activity']
            )
            self.transition_fallback.setdefault(key, row['transition'])

        group_keys = ['activity', 'object', 'object_type', 'higher_level_activity']
        grouped = raw_df.groupby(group_keys, dropna=False)

        n_trained, n_rejected, n_skipped = 0, 0, 0

        print(f"\n[SimModeller] Training WIP/RO-aware transition classifiers "
              f"(model_types={self.model_types})")

        for key_vals, group in grouped:
            key = tuple(
                str(k).strip() if pd.notna(k) else None
                for k in key_vals
            )

            if len(group) < _MIN_SAMPLES:
                n_skipped += 1
                continue

            fit_g, val_g = self._validation_split(group, self.val_size)
            X_fit = self._build_X(fit_g, feature_cols)
            X_val = self._build_X(val_g, feature_cols)
            X_all = self._build_X(group, feature_cols)

            y_fit_tr = fit_g['next_activity'].astype(str)
            y_val_tr = val_g['next_activity'].astype(str)
            y_all_tr = group['next_activity'].astype(str)

            if y_all_tr.nunique() < 2 or len(y_val_tr) == 0:
                n_skipped += 1
                continue

            le = LabelEncoder()
            le.fit(y_all_tr)
            y_enc_fit = le.transform(y_fit_tr)

            # ── statistical baseline accuracy on the SAME validation rows ──
            stat_dist = self.transition_fallback.get(key, {})
            if stat_dist:
                baseline_label = max(stat_dist, key=stat_dist.get)
            else:
                baseline_label = y_fit_tr.mode().iloc[0] if len(y_fit_tr) else None
            baseline_acc = (
                float((y_val_tr == baseline_label).mean())
                if baseline_label is not None else 0.0
            )

            best_acc, best_type = -1.0, None
            for mtype in self.model_types:
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        model = self._fit_classifier(mtype, X_fit, y_enc_fit)

                    known_mask = y_val_tr.isin(le.classes_)
                    if known_mask.sum() == 0:
                        continue
                    y_enc_val = le.transform(y_val_tr[known_mask])
                    y_pred    = model.predict(X_val[known_mask])
                    acc = accuracy_score(y_enc_val, y_pred)
                    if acc > best_acc:
                        best_acc, best_type = acc, mtype
                except Exception as exc:
                    print(f"    WIP-transition {key} [{mtype:>8s}]  FAILED: {exc}")

            if best_type is not None and best_acc > baseline_acc:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    y_enc_all = le.transform(y_all_tr)
                    final_model = self._fit_classifier(best_type, X_all, y_enc_all)
                self.transition_wip_models[key] = (final_model, le, feature_cols, best_type)
                n_trained += 1
                print(f"    ✓ {key}: {best_type}  "
                      f"(val Acc={best_acc:.3f} vs baseline={baseline_acc:.3f})")
            else:
                n_rejected += 1
                print(f"    ✗ {key}: best ML Acc={best_acc:.3f} did not beat "
                      f"baseline={baseline_acc:.3f} — falling back to statistical.")

        print(
            f"[SimModeller] WIP-transition training complete – "
            f"{n_trained} kept, {n_rejected} fell back to statistical baseline, "
            f"{n_skipped} skipped (< {_MIN_SAMPLES} samples or single-class)."
        )

    def predict_transitions_wip_aware(
        self,
        activity: str,
        object_name: str,
        object_type: str,
        higher_level_activity,
        object_attributes: dict,
        wip: float = 0.0,
        ro: float = 0.0,
    ) -> dict | None:
        """
        Predict transition probabilities using the WIP/RO-aware classifier
        trained by ``train_wip_transitions``.

        Returns ``None`` when no model was kept for this key (either too few
        samples, or it didn't beat the statistical baseline) — the caller
        should fall back to the frequency-based statistical distribution.
        """
        key = self._make_key(activity, object_name, object_type,
                             higher_level_activity)

        if key not in self.transition_wip_models:
            return None

        tr_model, le, feature_cols, _mtype = self.transition_wip_models[key]
        if not hasattr(tr_model, 'predict_proba'):
            return None

        features = self._attrs_to_features(object_attributes, feature_cols,
                                           wip=wip, ro=ro)
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

        # ── numeric context columns ───────────────────────────────────────
        for col in ('activity_index', 'hour_of_day', 'day_of_week'):
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
        activity_index: int = 0,
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

        # ── numeric context features ──────────────────────────────────────
        if 'activity_index' in global_feat_cols:
            features['activity_index'] = activity_index

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

    def predict_duration_wip_aware(
        self,
        activity: str,
        object_name: str,
        object_type: str,
        higher_level_activity,
        object_attributes: dict,
        wip: float = 0.0,
        ro: float = 0.0,
    ) -> float | None:
        """
        Point-estimate duration prediction (no noise) using the same
        per-key duration model as ``predict_duration_median``, but with
        live ``wip``/``ro`` values (from a ``LoadProfile`` lookup at
        simulation time) injected into the feature vector instead of the
        training-time defaults of 0.

        Returns ``None`` when no ML model is available for the given key.
        """
        key = self._make_key(activity, object_name, object_type,
                             higher_level_activity)

        if key not in self.duration_models:
            return None

        dur_model, feature_cols, _mtype = self.duration_models[key]
        features = self._attrs_to_features(object_attributes, feature_cols,
                                           wip=wip, ro=ro)
        X = pd.DataFrame([features])[feature_cols]
        for col in X.columns:
            X[col] = pd.to_numeric(X[col], errors='coerce')
        X = X.fillna(0.0)

        median_pred = float(dur_model.predict(X)[0])
        return max(0.1, median_pred)

    def predict_waiting_time(
        self,
        activity: str,
        object_name: str,
        object_type: str,
        higher_level_activity,
        object_attributes: dict,
        wip: float = 0.0,
        ro: float = 0.0,
    ) -> float | None:
        """
        Predict a sampled waiting time (minutes) — the queueing/contention
        gap before ``activity`` starts — analogous to ``predict_duration``
        but for the ``waiting_models`` family.  ``wip``/``ro`` come from a
        live ``LoadProfile`` lookup at simulation time.

        Returns ``None`` when no waiting-time model is available for the
        given key, signalling the caller to fall back to a fixed/no gap.
        """
        key = self._make_key(activity, object_name, object_type,
                             higher_level_activity)

        if key not in self.waiting_models:
            return None

        wait_model, feature_cols, _mtype = self.waiting_models[key]
        features = self._attrs_to_features(object_attributes, feature_cols,
                                           wip=wip, ro=ro)
        X = pd.DataFrame([features])[feature_cols]
        for col in X.columns:
            X[col] = pd.to_numeric(X[col], errors='coerce')
        X = X.fillna(0.0)

        median_pred = float(wait_model.predict(X)[0])

        std_pred = 0.0
        if key in self.waiting_std_models:
            std_model, _, _ = self.waiting_std_models[key]
            std_pred = max(0.0, float(std_model.predict(X)[0]))

        sampled = np.random.normal(median_pred, std_pred) if std_pred > 0 else median_pred
        return max(0.0, float(sampled))

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

        if not hasattr(tr_model, 'predict_proba'):
            return None

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
                           feature_cols: list[str],
                           wip: float = 0.0,
                           ro: float = 0.0) -> dict:
        """Map ``object_attributes`` dict to the ``attr_*`` / ``prev_*``
        feature space.  ``prev_*`` columns default to 0 here and are
        overridden by the caller when activity history is available.
        ``wip``/``ro`` come from a live ``LoadProfile`` lookup at
        simulation time (they aren't part of ``object_attributes``)."""
        feats = {}
        for col in feature_cols:
            if col.startswith('attr_'):
                feats[col] = object_attributes.get(col[5:], 0)
            elif col == 'wip':
                feats[col] = wip
            elif col == 'ro':
                feats[col] = ro
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
