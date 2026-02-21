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
# Model factories (default parameters)
# ═══════════════════════════════════════════════════════════════════════════════

def _default_regressor(model_type: str, random_state: int = 42):
    """Return a regressor instance with sensible defaults."""
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
    and the **best one on the temporal test set** is automatically selected.

    Parameters
    ----------
    model_types : list[str]
        List of model types to train and compare.
        Supported: ``'xgboost'``, ``'linear'``, ``'lasso'``, ``'mlp'``.
    optimize_hyperparams : bool
        If ``True``, use Optuna to find optimal hyper-parameters on the
        training split.  Requires ``optuna`` to be installed.
    n_optuna_trials : int
        Number of Optuna trials per model (only used when
        *optimize_hyperparams* is True).
    test_size : float
        Fraction of data reserved for the temporal test set (default 0.30).
    random_state : int
        Random seed for reproducibility.
    """

    def __init__(
        self,
        model_types: list[str] | None = None,
        optimize_hyperparams: bool = False,
        n_optuna_trials: int = 50,
        test_size: float = 0.30,
        train_transitions: bool = True,
        random_state: int = 42,
    ):
        self.model_types = model_types or ['xgboost']
        self.train_transitions = train_transitions
        self.optimize_hyperparams  = optimize_hyperparams
        self.n_optuna_trials       = n_optuna_trials
        self.test_size             = test_size
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

        # ── evaluation metrics ────────────────────────────────────────────
        self.duration_test_metrics:   dict = {}   # key -> {mae, rmse, r2, model_type}
        self.transition_test_metrics: dict = {}   # key -> {accuracy, f1, model_type}
        # All model results (for comparison reporting)
        self.duration_all_results:   dict = {}   # key -> {model_type: {mae, rmse, r2}}
        self.transition_all_results: dict = {}   # key -> {model_type: {accuracy, f1}}

        self._trained = False

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
        return [c for c in df.columns if c.startswith('attr_')]

    @staticmethod
    def _build_X(group: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
        """Return numeric feature matrix (impute missing values with 0)."""
        X = group[feature_cols].copy()
        for col in feature_cols:
            X[col] = pd.to_numeric(X[col], errors='coerce')
        return X.fillna(0.0)

    # ------------------------------------------------------------------
    # Temporal split helper
    # ------------------------------------------------------------------

    @staticmethod
    def _temporal_split(group: pd.DataFrame, test_size: float = 0.30):
        """
        Sort *group* by ``timestamp_start`` and split into train / test.
        The first ``1 - test_size`` fraction becomes training data.
        """
        if 'timestamp_start' not in group.columns:
            n_test = max(1, int(len(group) * test_size))
            return group.iloc[:-n_test], group.iloc[-n_test:]

        sorted_g = group.sort_values('timestamp_start').reset_index(drop=True)
        split_idx = int(len(sorted_g) * (1 - test_size))
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

        For each activity key, every model type in ``self.model_types`` is
        trained.  The best model (lowest MAE for duration, highest accuracy
        for transitions) on the temporal test set is selected automatically.

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

        feature_cols = self._feature_cols(raw_df)
        if not feature_cols:
            print("[SimModeller] No attr_* columns found – ML models skipped, "
                  "using statistical fallbacks.")
            self._trained = True
            return

        group_keys = ['activity', 'object', 'object_type',
                      'higher_level_activity']
        grouped = raw_df.groupby(group_keys, dropna=False)

        dur_trained = 0
        tr_trained  = 0

        print(f"\n[SimModeller] Model types      : {self.model_types}")
        print(f"[SimModeller] Optuna tuning    : {self.optimize_hyperparams}")
        print(f"[SimModeller] Temporal test %  : {self.test_size:.0%}")
        print(f"[SimModeller] Feature columns  : {feature_cols}")

        for key_vals, group in grouped:
            key = tuple(
                str(k).strip() if pd.notna(k) else None
                for k in key_vals
            )

            if len(group) < _MIN_SAMPLES:
                continue

            # ── temporal train / test split ───────────────────────────────
            train_g, test_g = self._temporal_split(group, self.test_size)

            X_train = self._build_X(train_g, feature_cols)
            X_test  = self._build_X(test_g,  feature_cols)

            # skip if all features are constant (no signal)
            if (X_train.nunique() <= 1).all():
                continue

            print(f"\n  Key: {key}")
            print(f"    Train size: {len(train_g)}, Test size: {len(test_g)}")

            # ==============================================================
            # 1. Duration model — train ALL types, pick best by MAE
            # ==============================================================
            y_train_dur = train_g['duration'].astype(float)
            y_test_dur  = test_g['duration'].astype(float)

            best_dur_mae   = float('inf')
            best_dur_model = None
            best_dur_type  = None
            best_dur_metrics = None
            dur_results = {}

            for mtype in self.model_types:
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        model = self._fit_regressor(mtype, X_train, y_train_dur)

                    # evaluate on test
                    if len(X_test) > 0:
                        y_pred = model.predict(X_test)
                        mae  = mean_absolute_error(y_test_dur, y_pred)
                        rmse = np.sqrt(mean_squared_error(y_test_dur, y_pred))
                        r2   = r2_score(y_test_dur, y_pred)
                        dur_results[mtype] = {
                            'mae': mae, 'rmse': rmse, 'r2': r2,
                        }
                        print(f"    Duration [{mtype:>8s}]  "
                              f"MAE={mae:.3f}  RMSE={rmse:.3f}  R²={r2:.3f}")

                        if mae < best_dur_mae:
                            best_dur_mae     = mae
                            best_dur_model   = model
                            best_dur_type    = mtype
                            best_dur_metrics = dur_results[mtype]
                    else:
                        # No test data — keep first successful model
                        if best_dur_model is None:
                            best_dur_model = model
                            best_dur_type  = mtype

                except Exception as exc:
                    print(f"    Duration [{mtype:>8s}]  FAILED: {exc}")

            if best_dur_model is not None:
                self.duration_models[key] = (best_dur_model, feature_cols,
                                             best_dur_type)
                dur_trained += 1
                if best_dur_metrics:
                    self.duration_test_metrics[key] = {
                        **best_dur_metrics, 'model_type': best_dur_type,
                    }
                if dur_results:
                    self.duration_all_results[key] = dur_results

                print(f"    ✓ Best duration model: {best_dur_type}"
                      + (f"  (MAE={best_dur_mae:.3f})"
                         if best_dur_mae < float('inf') else ""))

                # ── Duration std model (on abs residuals) — same type ─────
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        residuals = np.abs(
                            y_train_dur.values
                            - best_dur_model.predict(X_train)
                        )
                        std_model = self._fit_regressor(
                            best_dur_type, X_train, pd.Series(residuals)
                        )
                        self.duration_std_models[key] = (std_model,
                                                         feature_cols,
                                                         best_dur_type)
                except Exception as exc:
                    print(f"    [!] Duration std model failed: {exc}")

            # ==============================================================
            # 2. Transition model — train ALL types, pick best by accuracy
            #    (skipped when train_transitions=False, e.g. ml_duration_only)
            # ==============================================================
            if not self.train_transitions:
                continue

            y_train_tr = train_g['next_activity'].astype(str)
            y_test_tr  = test_g['next_activity'].astype(str)

            if y_train_tr.nunique() < 2:
                continue

            le = LabelEncoder()
            y_enc_train = le.fit_transform(y_train_tr)

            best_tr_acc   = -1.0
            best_tr_model = None
            best_tr_type  = None
            best_tr_metrics = None
            tr_results = {}

            for mtype in self.model_types:
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        model = self._fit_classifier(
                            mtype, X_train, y_enc_train
                        )

                    # evaluate on test
                    if len(X_test) > 0:
                        known_mask = y_test_tr.isin(le.classes_)
                        if known_mask.sum() > 0:
                            y_enc_test = le.transform(y_test_tr[known_mask])
                            y_pred_tr  = model.predict(X_test[known_mask])
                            acc = accuracy_score(y_enc_test, y_pred_tr)
                            f1  = f1_score(y_enc_test, y_pred_tr,
                                           average='weighted',
                                           zero_division=0)
                            tr_results[mtype] = {
                                'accuracy': acc, 'f1': f1,
                            }
                            print(f"    Transition [{mtype:>8s}]  "
                                  f"Acc={acc:.3f}  F1={f1:.3f}")

                            if acc > best_tr_acc:
                                best_tr_acc     = acc
                                best_tr_model   = model
                                best_tr_type    = mtype
                                best_tr_metrics = tr_results[mtype]
                        else:
                            if best_tr_model is None:
                                best_tr_model = model
                                best_tr_type  = mtype
                    else:
                        if best_tr_model is None:
                            best_tr_model = model
                            best_tr_type  = mtype

                except Exception as exc:
                    print(f"    Transition [{mtype:>8s}]  FAILED: {exc}")

            if best_tr_model is not None:
                self.transition_models[key] = (best_tr_model, le,
                                               feature_cols, best_tr_type)
                tr_trained += 1
                if best_tr_metrics:
                    self.transition_test_metrics[key] = {
                        **best_tr_metrics, 'model_type': best_tr_type,
                    }
                if tr_results:
                    self.transition_all_results[key] = tr_results

                print(f"    ✓ Best transition model: {best_tr_type}"
                      + (f"  (Acc={best_tr_acc:.3f})"
                         if best_tr_acc >= 0 else ""))

        print(
            f"\n[SimModeller] Training complete – "
            f"{dur_trained} duration models, {tr_trained} transition models."
        )
        self._trained = True

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
    ) -> float | None:
        """
        Return the raw ML duration prediction **without** adding
        ML-predicted noise (std).

        Used by ``'ml_duration_only'`` mode, where the caller adds statistical
        std from the extracted data instead.

        Returns ``None`` when no ML model is available for the given key.
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
        return max(0.1, median_pred)

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
        """Map ``object_attributes`` dict to the ``attr_*`` feature space."""
        return {
            col: object_attributes.get(col[5:], 0)   # strip 'attr_' prefix
            for col in feature_cols
        }

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

        if self.duration_test_metrics:
            lines.append("\n  ── Best duration model per key (on test set) ──")
            for key, m in self.duration_test_metrics.items():
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
                        key in self.duration_test_metrics
                        and self.duration_test_metrics[key]['model_type'] == mtype
                    ) else ""
                    lines.append(
                        f"      {mtype:>8s}:  MAE={m['mae']:.3f}  "
                        f"RMSE={m['rmse']:.3f}  R²={m['r2']:.3f}{best_marker}"
                    )

        if self.transition_test_metrics:
            lines.append("\n  ── Best transition model per key (on test set) ──")
            for key, m in self.transition_test_metrics.items():
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
                        key in self.transition_test_metrics
                        and self.transition_test_metrics[key]['model_type'] == mtype
                    ) else ""
                    lines.append(
                        f"      {mtype:>8s}:  Acc={m['accuracy']:.3f}  "
                        f"F1={m['f1']:.3f}{best_marker}"
                    )

        return "\n".join(lines)
