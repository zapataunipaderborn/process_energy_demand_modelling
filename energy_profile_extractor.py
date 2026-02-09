"""
EnergyProfileExtractor
======================

Reusable class to extract energy profiles from time-series data,
train ML models that learn the profile shapes, and predict new
profiles for arbitrary output lengths.

Usage
-----
    from utils.energy_profile_extractor import EnergyProfileExtractor

    extractor = EnergyProfileExtractor(
        value_column='heating_power',
        case_id_column='case_id',
        activity_column='activity',
        start_end_column='start_end',
        timestamp_column='timestamp_start',
        attribute_columns_categorical=['recipe'],
        attribute_columns_numerical=[],
    )

    # Fit on training data (a DataFrame with time-series curves)
    results_df = extractor.fit(df_train_combined)

    # Predict a single curve
    curve = extractor.predict_curve(activity='autoclave_1', output_length=120, recipe='A')

    # Predict curves for a full event log
    df_predicted = extractor.predict_event_log(df_test, points_per_second=0.1)
"""

from __future__ import annotations

import warnings
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.signal import medfilt
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

try:
    from xgboost import XGBRegressor

    _HAS_XGB = True
except ImportError:
    _HAS_XGB = False


class EnergyProfileExtractor:
    """Train and predict energy profiles from time-series event data.

    The pipeline works as follows:

    1. **Extract curves** – group data by ``case_id_column``, detect
       start / end markers, and extract the value column as raw curves.
    2. **Resample** – bring every curve to a fixed *latent* length so
       that a single regression model can be trained.
    3. **Train** – build a regression dataset
       ``(curve_index, activity, …attributes) → value`` and train one
       or more models.
    4. **Predict** – given an activity and attribute values, predict the
       latent curve and inverse-resample to any requested output length.
    """

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(
        self,
        value_column: str = "heating_power",
        case_id_column: str = "case_id",
        activity_column: str = "activity",
        start_end_column: str = "start_end",
        timestamp_column: str = "timestamp_start",
        attribute_columns_categorical: Optional[List[str]] = None,
        attribute_columns_numerical: Optional[List[str]] = None,
        test_size: float = 0.2,
        random_state: int = 42,
        models: Optional[Dict[str, Any]] = None,
        verbose: bool = True,
    ):
        self.value_column = value_column
        self.case_id_column = case_id_column
        self.activity_column = activity_column
        self.start_end_column = start_end_column
        self.timestamp_column = timestamp_column
        self.attribute_columns_categorical = attribute_columns_categorical or []
        self.attribute_columns_numerical = attribute_columns_numerical or []
        self.test_size = test_size
        self.random_state = random_state
        self.verbose = verbose

        # models supplied by the user or defaults
        self._user_models = models
        self._default_models = self._build_default_models()

        # Fitted state --------------------------------------------------
        self.target_length_: Optional[int] = None
        self.curves_: Optional[List[dict]] = None
        self.results_df_: Optional[pd.DataFrame] = None
        self.trained_models_: Dict[str, Pipeline] = {}
        self.best_model_: Optional[Pipeline] = None
        self.best_model_name_: Optional[str] = None

    # ------------------------------------------------------------------
    # Default models
    # ------------------------------------------------------------------

    @staticmethod
    def _build_default_models() -> Dict[str, Any]:
        models: Dict[str, Any] = {
            "Linear Regression": LinearRegression(),
        }
        if _HAS_XGB:
            models["XGBoost"] = XGBRegressor(
                n_estimators=300,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                n_jobs=-1,
            )
        return models

    # ------------------------------------------------------------------
    # Column helpers
    # ------------------------------------------------------------------

    @property
    def _all_attribute_columns(self) -> List[str]:
        return self.attribute_columns_categorical + self.attribute_columns_numerical

    @property
    def _feature_columns(self) -> List[str]:
        return (
            ["curve_index", self.activity_column]
            + self.attribute_columns_categorical
            + self.attribute_columns_numerical
        )

    # ------------------------------------------------------------------
    # Curve extraction
    # ------------------------------------------------------------------

    def _extract_curves(self, df: pd.DataFrame) -> Tuple[List[dict], int]:
        """Extract individual value curves from the DataFrame."""
        curves: List[dict] = []
        lengths: List[int] = []

        for case_id, group in df.groupby(self.case_id_column):
            group = group.sort_values(self.timestamp_column).reset_index(drop=True)

            start_idxs = group[group[self.start_end_column] == "start"].index
            end_idxs = group[group[self.start_end_column] == "end"].index

            for s in start_idxs:
                ends = end_idxs[end_idxs > s]
                if len(ends) == 0:
                    continue

                e = ends[0]
                curve_df = group.iloc[s : e + 1]

                if len(curve_df) < 3:
                    continue

                values = curve_df[self.value_column].values.astype(float)

                curve_dict: dict = {
                    "case_id": case_id,
                    self.activity_column: curve_df.iloc[0][self.activity_column],
                    "values": values,
                }
                for col in self._all_attribute_columns:
                    curve_dict[col] = curve_df.iloc[0][col]

                curves.append(curve_dict)
                lengths.append(len(values))

        if not curves:
            raise ValueError("No valid curves found in the data.")

        target_length = int(np.median(lengths))
        return curves, target_length

    # ------------------------------------------------------------------
    # Resampling
    # ------------------------------------------------------------------

    @staticmethod
    def resample_curve(values: np.ndarray, target_len: int) -> np.ndarray:
        """Resample a curve to *target_len* with outlier clipping + median filter."""
        values = np.asarray(values, dtype=float)
        mean, std = values.mean(), values.std()
        if std > 0:
            values = np.clip(values, mean - 3 * std, mean + 3 * std)
        kernel = min(5, len(values))
        if kernel % 2 == 0:
            kernel = max(kernel - 1, 1)
        values = medfilt(values, kernel_size=kernel)
        x_old = np.linspace(0, 1, len(values))
        x_new = np.linspace(0, 1, target_len)
        return np.interp(x_new, x_old, values)

    @staticmethod
    def inverse_resample_curve(values: np.ndarray, new_len: int) -> np.ndarray:
        """Resample a fixed-length predicted curve to *new_len*."""
        x_old = np.linspace(0, 1, len(values))
        x_new = np.linspace(0, 1, new_len)
        return np.interp(x_new, x_old, values)

    # ------------------------------------------------------------------
    # Building the regression dataset
    # ------------------------------------------------------------------

    def _curves_to_dataframe(self, curves: List[dict]) -> pd.DataFrame:
        rows: List[dict] = []
        for curve_id, c in enumerate(curves):
            for idx, y in enumerate(c["values"], start=1):
                row = {
                    "curve_id": curve_id,
                    "curve_index": idx,
                    self.activity_column: c[self.activity_column],
                    "y": float(y),
                }
                for col in self._all_attribute_columns:
                    row[col] = c[col]
                rows.append(row)
        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # Preprocessing pipeline
    # ------------------------------------------------------------------

    def _build_preprocessor(self) -> ColumnTransformer:
        return ColumnTransformer(
            transformers=[
                (
                    "cat",
                    OneHotEncoder(handle_unknown="ignore"),
                    [self.activity_column] + self.attribute_columns_categorical,
                ),
                (
                    "num",
                    "passthrough",
                    ["curve_index"] + self.attribute_columns_numerical,
                ),
            ]
        )

    # ------------------------------------------------------------------
    # fit
    # ------------------------------------------------------------------

    def fit(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fit the extractor on a combined time-series DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            Must contain at least the columns specified during
            construction (value, case_id, activity, start_end,
            timestamp, and attribute columns).

        Returns
        -------
        pd.DataFrame
            Results table with RMSE and R² per model, sorted ascending
            by RMSE.
        """
        # 1. Extract raw curves
        curves, target_length = self._extract_curves(df)
        self.target_length_ = target_length
        if self.verbose:
            print(f"Extracted {len(curves)} curves  |  latent length = {target_length}")

        # 2. Resample all to latent size
        for c in curves:
            c["values"] = self.resample_curve(c["values"], target_length)
        self.curves_ = curves

        # 3. Build regression dataset
        df_model = self._curves_to_dataframe(curves)
        if self.verbose:
            print(f"Regression samples: {len(df_model)}")

        # 4. Train / test split (by curve)
        curve_ids = df_model["curve_id"].unique()
        train_ids, test_ids = train_test_split(
            curve_ids, test_size=self.test_size, random_state=self.random_state
        )
        train_df = df_model[df_model["curve_id"].isin(train_ids)]
        test_df = df_model[df_model["curve_id"].isin(test_ids)]

        X_train = train_df[self._feature_columns]
        y_train = train_df["y"].values
        X_test = test_df[self._feature_columns]
        y_test = test_df["y"].values

        # 5. Preprocessing
        preprocess = self._build_preprocessor()

        # 6. Train models
        models = self._user_models if self._user_models else self._default_models
        results: List[dict] = []
        self.trained_models_ = {}

        for name, reg in models.items():
            if self.verbose:
                print(f"  Training {name} …")

            pipe = Pipeline([("prep", preprocess), ("model", reg)])
            pipe.fit(X_train, y_train)
            self.trained_models_[name] = pipe

            y_pred = pipe.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)
            rmse = float(np.sqrt(mse))
            r2 = float(r2_score(y_test, y_pred))

            results.append({"Model": name, "RMSE": rmse, "R2": r2})
            if self.verbose:
                print(f"    RMSE={rmse:.4f}  R²={r2:.4f}")

        self.results_df_ = pd.DataFrame(results).sort_values("RMSE").reset_index(drop=True)
        best_name = self.results_df_.iloc[0]["Model"]
        self.best_model_ = self.trained_models_[best_name]
        self.best_model_name_ = best_name

        if self.verbose:
            print(f"\nBest model: {best_name}")

        return self.results_df_

    # ------------------------------------------------------------------
    # predict_curve
    # ------------------------------------------------------------------

    def predict_curve(
        self,
        activity: str,
        output_length: int,
        model: Optional[Pipeline] = None,
        **attributes: Any,
    ) -> np.ndarray:
        """Predict a single energy-profile curve.

        Parameters
        ----------
        activity : str
            Activity label (e.g. ``'autoclave_1'``).
        output_length : int
            Desired number of points in the output curve.
        model : Pipeline, optional
            Specific trained model to use. Defaults to the best model
            found during :meth:`fit`.
        **attributes
            Additional attribute values, e.g. ``recipe='A'``.

        Returns
        -------
        np.ndarray
            Predicted curve of length *output_length*.
        """
        if self.target_length_ is None:
            raise RuntimeError("Call .fit() before .predict_curve()")

        model = model or self.best_model_

        X_latent = pd.DataFrame(
            {
                "curve_index": np.arange(1, self.target_length_ + 1),
                self.activity_column: activity,
                **attributes,
            }
        )

        latent_curve = model.predict(X_latent)
        return self.inverse_resample_curve(latent_curve, output_length)

    # ------------------------------------------------------------------
    # predict_event_log  –  predict curves for every row in an event log
    # ------------------------------------------------------------------

    def predict_event_log(
        self,
        df_event_log: pd.DataFrame,
        points_per_second: float = 0.1,
        activity_filter: Optional[List[str]] = None,
        model: Optional[Pipeline] = None,
    ) -> pd.DataFrame:
        """Generate predicted energy curves for an entire event log.

        Parameters
        ----------
        df_event_log : pd.DataFrame
            Event log with columns: *case_id*, *activity*,
            *timestamp_start*, *timestamp_end*, plus any attribute
            columns.
        points_per_second : float
            Temporal resolution of the output curves.
        activity_filter : list[str], optional
            If given, only predict curves for these activities.
        model : Pipeline, optional
            Defaults to the best model.

        Returns
        -------
        pd.DataFrame
            A long-format DataFrame with predicted curves joined back
            to event metadata.
        """
        if self.target_length_ is None:
            raise RuntimeError("Call .fit() before .predict_event_log()")

        model = model or self.best_model_
        df = df_event_log.copy()

        if activity_filter:
            df = df[df[self.activity_column].isin(activity_filter)]
        df = df.sort_values(by=[self.timestamp_column, self.case_id_column]).reset_index(
            drop=True
        )

        list_df: List[pd.DataFrame] = []

        for _, row in df.iterrows():
            duration_seconds = (
                row["timestamp_end"] - row[self.timestamp_column]
            ).total_seconds()
            output_length = max(3, int(duration_seconds * points_per_second))

            # Gather attribute kwargs
            attrs: dict = {}
            for col in self._all_attribute_columns:
                if col in row.index:
                    attrs[col] = row[col]

            curve_values = self.predict_curve(
                activity=row[self.activity_column],
                output_length=output_length,
                model=model,
                **attrs,
            )

            timestamps = pd.date_range(
                start=row[self.timestamp_column],
                end=row["timestamp_end"],
                periods=output_length,
            )

            df_curve = pd.DataFrame(
                {
                    self.timestamp_column: timestamps,
                    self.value_column: curve_values,
                }
            )
            df_curve[self.start_end_column] = "in_between"
            df_curve.loc[df_curve.index[0], self.start_end_column] = "start"
            df_curve.loc[df_curve.index[-1], self.start_end_column] = "end"
            df_curve[self.activity_column] = row[self.activity_column]
            df_curve[self.case_id_column] = row[self.case_id_column]
            df_curve = df_curve.reset_index()
            df_curve["length"] = df_curve["index"].max()
            list_df.append(df_curve)

        if not list_df:
            warnings.warn("No curves predicted – check activity_filter or data.")
            return pd.DataFrame()

        result = (
            pd.concat(list_df, ignore_index=True)
            .sort_values(by=[self.timestamp_column])
            .reset_index(drop=True)
        )

        # Forward-fill event-log metadata
        result = result.merge(
            df_event_log,
            on=[self.timestamp_column, self.activity_column, self.case_id_column],
            how="left",
        )
        result = result.ffill()
        result = result.sort_values(by=[self.timestamp_column, self.case_id_column])

        return result
