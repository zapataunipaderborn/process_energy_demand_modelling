"""
sim_modeller.py
===============
Trains XGBoost models that power the 'ml' simulation mode.

Three model families are trained, all keyed by
``(activity, object, object_type, higher_level_activity)``:

1. **duration_models**     – XGBRegressor predicting *median* duration from
                              ``object_attributes`` (objective=``reg:absoluteerror``).
2. **duration_std_models** – XGBRegressor predicting *median absolute deviation*
                              of duration (fitted on absolute residuals of model 1).
3. **transition_models**   – XGBClassifier predicting the *next activity*
                              (including ``'__END__'``) from
                              ``object_attributes``.

When attributes carry no useful signal (< 5 training samples, or all attribute
values are constant) the ``SimModeller`` returns ``None`` from its predict
methods, which tells the simulation to fall back to the statistical mode.
"""

from __future__ import annotations

import warnings
import numpy as np
import pandas as pd

try:
    from xgboost import XGBRegressor, XGBClassifier
    _XGBOOST_AVAILABLE = True
except ImportError:
    _XGBOOST_AVAILABLE = False
    warnings.warn(
        "xgboost is not installed – ML mode will always fall back to statistical mode. "
        "Install it with:  pip install xgboost",
        ImportWarning,
        stacklevel=2,
    )

from sklearn.preprocessing import LabelEncoder

# Minimum number of training samples required before fitting a model.
_MIN_SAMPLES = 5


class SimModeller:
    """
    Train and serve XGBoost models for the 'ml' simulation mode.

    Parameters
    ----------
    n_estimators : int
        Number of trees in each XGBoost model.
    max_depth : int
        Maximum tree depth.
    random_state : int
        Random seed for reproducibility.
    """

    def __init__(self, n_estimators: int = 100, max_depth: int = 4,
                 random_state: int = 42):
        self.n_estimators = n_estimators
        self.max_depth    = max_depth
        self.random_state = random_state

        # ── model stores ──────────────────────────────────────────────────
        self.duration_models:     dict = {}   # key -> (XGBRegressor, feature_cols)
        self.duration_std_models: dict = {}   # key -> (XGBRegressor, feature_cols)
        self.transition_models:   dict = {}   # key -> (XGBClassifier, LabelEncoder, feature_cols)

        # ── statistical fallbacks (populated from stats_df during train) ──
        self.duration_fallback:   dict = {}   # key -> (dist_name, dist_params)
        self.transition_fallback: dict = {}   # key -> transition dict

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
        # Convert to numeric – non-numeric map to NaN then fill 0
        for col in feature_cols:
            X[col] = pd.to_numeric(X[col], errors='coerce')
        return X.fillna(0.0)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def train(self, raw_df: pd.DataFrame, stats_df: pd.DataFrame) -> None:
        """
        Train all ML models.

        Parameters
        ----------
        raw_df : pd.DataFrame
            Output from ``sim_extractor.extract_process`` – one row per
            activity instance with columns ``activity``, ``object``,
            ``object_type``, ``higher_level_activity``, ``duration``,
            ``next_activity``, and ``attr_*``.
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

        if not _XGBOOST_AVAILABLE:
            print("[SimModeller] xgboost not available – only statistical fallbacks stored.")
            self._trained = True
            return

        feature_cols = self._feature_cols(raw_df)
        if not feature_cols:
            print("[SimModeller] No attr_* columns found – ML models skipped, using statistical fallbacks.")
            self._trained = True
            return

        group_keys = ['activity', 'object', 'object_type', 'higher_level_activity']
        grouped = raw_df.groupby(group_keys, dropna=False)

        dur_trained = 0
        tr_trained  = 0

        for key_vals, group in grouped:
            key = tuple(
                str(k).strip() if pd.notna(k) else None
                for k in key_vals
            )

            if len(group) < _MIN_SAMPLES:
                continue

            X = self._build_X(group, feature_cols)

            # ── skip if all features are constant (no signal) ─────────────
            if (X.nunique() <= 1).all():
                continue

            # ── 1. Duration model ─────────────────────────────────────────
            y_dur = group['duration'].astype(float)
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    dur_model = XGBRegressor(
                        n_estimators=self.n_estimators,
                        max_depth=self.max_depth,
                        random_state=self.random_state,
                        objective='reg:absoluteerror',   # median (MAE)
                        verbosity=0,
                    )
                    dur_model.fit(X, y_dur)
                    self.duration_models[key] = (dur_model, feature_cols)
                    dur_trained += 1

                    # ── 2. Duration std model (on abs residuals) ──────────
                    residuals = np.abs(y_dur.values - dur_model.predict(X))
                    std_model = XGBRegressor(
                        n_estimators=self.n_estimators,
                        max_depth=self.max_depth,
                        random_state=self.random_state,
                        objective='reg:absoluteerror',   # median (MAE)
                        verbosity=0,
                    )
                    std_model.fit(X, residuals)
                    self.duration_std_models[key] = (std_model, feature_cols)
            except Exception as exc:
                print(f"[SimModeller] Duration model failed for {key}: {exc}")

            # ── 3. Transition model ───────────────────────────────────────
            y_tr = group['next_activity'].astype(str)
            unique_targets = y_tr.nunique()
            if unique_targets < 2:
                continue   # only one outcome – no model needed

            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    le = LabelEncoder()
                    y_enc = le.fit_transform(y_tr)
                    tr_model = XGBClassifier(
                        n_estimators=self.n_estimators,
                        max_depth=self.max_depth,
                        random_state=self.random_state,
                        verbosity=0,
                        eval_metric='mlogloss',
                        use_label_encoder=False,
                    )
                    tr_model.fit(X, y_enc)
                    self.transition_models[key] = (tr_model, le, feature_cols)
                    tr_trained += 1
            except Exception as exc:
                print(f"[SimModeller] Transition model failed for {key}: {exc}")

        print(
            f"[SimModeller] Training complete – "
            f"{dur_trained} duration models, {tr_trained} transition models trained."
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
        The prediction is centred on the **median** (reg:absoluteerror objective).

        Returns ``None`` when no ML model is available for the given key,
        signalling the caller to fall back to statistical mode.
        """
        key = self._make_key(activity, object_name, object_type, higher_level_activity)

        if key not in self.duration_models:
            return None

        dur_model, feature_cols = self.duration_models[key]
        features = self._attrs_to_features(object_attributes, feature_cols)
        X = pd.DataFrame([features])[feature_cols]
        for col in X.columns:
            X[col] = pd.to_numeric(X[col], errors='coerce')
        X = X.fillna(0.0)

        median_pred = float(dur_model.predict(X)[0])

        std_pred = 0.0
        if key in self.duration_std_models:
            std_model, _ = self.duration_std_models[key]
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
        Return the raw ML median duration prediction **without** adding
        ML-predicted noise (std).

        Used by ``'ml_duration_only'`` mode, where the caller adds statistical
        std from the extracted data instead.

        Returns ``None`` when no ML model is available for the given key,
        signalling the caller to fall back to statistical mode.
        """
        key = self._make_key(activity, object_name, object_type, higher_level_activity)

        if key not in self.duration_models:
            return None

        dur_model, feature_cols = self.duration_models[key]
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
        key = self._make_key(activity, object_name, object_type, higher_level_activity)

        if key not in self.transition_models:
            return None

        tr_model, le, feature_cols = self.transition_models[key]
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
    def _attrs_to_features(object_attributes: dict, feature_cols: list[str]) -> dict:
        """Map ``object_attributes`` dict to the ``attr_*`` feature space."""
        return {
            col: object_attributes.get(col[5:], 0)   # strip 'attr_' prefix
            for col in feature_cols
        }

    def summary(self) -> str:
        lines = [
            "SimModeller summary",
            f"  Duration models  : {len(self.duration_models)}",
            f"  Duration std mdls: {len(self.duration_std_models)}",
            f"  Transition models: {len(self.transition_models)}",
            f"  Fallback entries : {len(self.duration_fallback)} (duration),"
            f" {len(self.transition_fallback)} (transition)",
        ]
        return "\n".join(lines)
