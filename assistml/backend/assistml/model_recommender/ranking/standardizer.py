import pandas as pd
from typing import List, Any, Dict, Optional, Union


class Standardizer:
    """
    A generic standardizer that quantizes a set of values.
    It determines whether the values are numeric or categorical,
    and applies quantile‑based binning for numeric values or cleaning for categorical ones.
    It also provides methods for inverse transformation so that concrete values can be suggested.
    """
    def __init__(self, bins: int = 5, numeric_threshold: float = 0.8):
        """
        Parameters:
            bins (int): Number of bins to use for numeric values.
            numeric_threshold (float): Fraction of values that must be numeric
                                       to consider the hyperparameter as numeric.
        """
        self.bins = bins
        self.numeric_threshold = numeric_threshold
        self.is_numeric: Union[bool, None] = None
        self.is_integer: Optional[bool] = None
        self.bin_intervals = None  # Will store pandas IntervalIndex for numeric binning
        self.numeric_labels = None  # e.g., ["Q1", "Q2", ...]
        self.categorical_mapping: Dict[Any, str] = {}  # maps original -> standardized
        self.inverse_categorical_mapping: Dict[str, Any] = {}  # maps standardized -> representative value

    def _determine_numeric(self, values: List[Any]) -> bool:
        """
        Check whether most of the values can be interpreted as numeric.
        """
        total = len(values)
        numeric_count = 0
        for v in values:
            try:
                float(v)
                numeric_count += 1
            except (ValueError, TypeError):
                continue
        return (numeric_count / total if total > 0 else 0) >= self.numeric_threshold

    def _fit_numeric(self, values: List[float]) -> None:
        """
        Fit the standardizer for numeric values by binning them.
        """
        series = pd.Series(values)
        if series.nunique() == 1:
            constant = series.iloc[0]
            self.bin_intervals = pd.IntervalIndex.from_tuples([(constant, constant)])
            self.numeric_labels = ["Q1"]
        else:
            binned = pd.qcut(series, q=self.bins, duplicates='drop', precision=10)
            self.bin_intervals = binned.cat.categories
            self.numeric_labels = [f"Q{i + 1}" for i in range(len(self.bin_intervals))]

    def _fit_categorical(self, values: List[Any]) -> None:
        """
        Fit the standardizer for categorical values by cleaning them.
        """
        standardized = [str(v).strip().lower() for v in values]
        self.categorical_mapping = {v: str(v).strip().lower() for v in values}
        freq: Dict[str, List[Any]] = {}
        for orig, std in zip(values, standardized):
            freq.setdefault(std, []).append(orig)
        self.inverse_categorical_mapping = {
            std: max(set(orig_list), key=orig_list.count)  # most frequent original
            for std, orig_list in freq.items()
        }

    def fit(self, values: List[Any]) -> None:
        """
        Fit the standardizer to a list of values.

        For numeric values, it will perform binning.
        For categorical values, it builds a mapping.
        """
        self.is_numeric = self._determine_numeric(values)
        if self.is_numeric:
            # Convert all values to float and keep only the numeric ones
            numeric_values = []
            outliers = []
            for v in values:
                try:
                    numeric_values.append(float(v))
                except (ValueError, TypeError):
                    outliers.append(v)
            self.is_integer = all(v.is_integer() for v in numeric_values)
            self._fit_numeric(numeric_values)
            if outliers:
                self._fit_categorical(outliers)
        else:
            self._fit_categorical(values)

    def _transform_numeric(self, values: List[float]) -> List[str]:
        """
        Transform numeric values into their corresponding bin labels.
        """
        series = pd.Series(values)
        bins = [interval.left for interval in self.bin_intervals] + [self.bin_intervals[-1].right]
        binned = pd.cut(series, bins=bins, include_lowest=True, labels=self.numeric_labels, precision=10)
        return binned.astype(str).tolist()

    def _transform_categorical(self, values: List[Any]) -> List[str]:
        """
        Transform categorical values using the stored mapping.
        """
        return [self.categorical_mapping.get(v, str(v).strip().lower()) for v in values]

    def transform(self, values: Union[Any, List[Any]]) -> Union[str, List[str]]:
        """
        Transform the given values into standardized (discrete) representations.

        Accepts a single value or a list of values.
        For numeric values, returns bin labels (e.g., "Q1").
        For categorical values, returns the cleaned string.
        """
        if self.is_numeric is None:
            raise ValueError("You must call fit() before transform().")

        if not isinstance(values, list):
            return self.transform([values])[0]

        if not self.is_numeric:
            return self._transform_categorical(values)

        numerics, outliers = [], []
        for idx, val in enumerate(values):
            try:
                numerics.append((idx, float(val)))
            except (ValueError, TypeError):
                outliers.append((idx, val))

        transformed: List[Union[str, None]] = [None] * len(values)

        if numerics:
            indices, nums = zip(*numerics)
            transformed_nums = self._transform_numeric(nums)
            for i, val in zip(indices, transformed_nums):
                transformed[i] = val

        if outliers:
            indices, cats = zip(*outliers)
            transformed_cats = self._transform_categorical(cats)
            for i, val in zip(indices, transformed_cats):
                transformed[i] = val

        return transformed

    def _inverse_transform_numeric(self, transformed: List[str]) -> List[Union[float, int]]:
        """
        Inverse transform numeric bin labels back to a representative value (the midpoint).
        """
        if len(self.numeric_labels) == 1:
            constant = (self.bin_intervals[0].left + self.bin_intervals[0].right) / 2
            return [constant for _ in transformed]
        rep_values = {
            label: (interval.left + interval.right) / 2 if not self.is_integer else round((interval.left + interval.right) / 2)
            for label, interval in zip(self.numeric_labels, self.bin_intervals)
        }
        return [rep_values.get(tv, None) for tv in transformed]

    def _inverse_transform_categorical(self, transformed: List[str]) -> List[Any]:
        """
        Inverse transform categorical standardized values back to a representative original value.
        """
        return [self.inverse_categorical_mapping.get(tv, tv) for tv in transformed]

    def inverse_transform(self, transformed: Union[str, List[str]]) -> Union[Any, List[Any]]:
        """
        Reverse the transformation.

        Accepts a single transformed value or a list of them.
        For numeric values, returns a representative numeric value (midpoint of bin).
        For categorical values, returns a representative original value.
        """
        if self.is_numeric is None:
            raise ValueError("You must call fit() before inverse_transform().")

        if not isinstance(transformed, list):
            return self.inverse_transform([transformed])[0]
        if self.is_numeric:
            return self._inverse_transform_numeric(transformed)
        else:
            return self._inverse_transform_categorical(transformed)
