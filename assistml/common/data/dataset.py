from enum import Enum
from typing import Dict, List, ForwardRef, Optional

import numpy as np
from beanie import Document, BackLink
from pydantic import Field
from pymongo import IndexModel

from .utils import CustomBaseModel, alias_generator

Task = ForwardRef("Task")
DatasetDescriptorNormalizer = ForwardRef("DatasetDescriptorNormalizer")


class TargetFeatureType(Enum):
    BINARY = "binary", "Binary"
    CATEGORICAL = "categorical", "Categorical"
    NUMERICAL = "numerical", "Numerical"

    def __new__(cls, value: str, display_name: str):
        self = object.__new__(cls)
        self._value_ = value
        self.display_name = display_name
        return self

    def __hash__(self):
        return hash(self.value)

class Info(CustomBaseModel):
    mlsea_uri: Optional[str] = None
    dataset_name: str
    target_label: str
    target_feature_type: TargetFeatureType
    observations: int
    analyzed_observations: int
    nr_analyzed_features: int
    nr_total_features: int
    numerical_ratio: float
    categorical_ratio: float
    datetime_ratio: float
    unstructured_ratio: float
    analyzed_features: list[str]
    discarded_features: list[str]
    analysis_time: float


class Quantiles(CustomBaseModel):
    q0: float
    q1: float
    q2: float
    q3: float
    q4: float
    iqr: float


class Outliers(CustomBaseModel):
    number: int
    actual_values: list[float]


class Distribution(CustomBaseModel):
    normal: bool
    exponential: bool


class NumericalFeature(CustomBaseModel):
    monotonous_filtering: float
    anova_f1: float
    anova_pvalue: float
    mutual_info: Optional[float] = None  # Does not exist for regression
    missing_values: int
    min_value: float
    max_value: float
    min_orderm: float
    max_orderm: float
    quartiles: Quantiles
    outliers: Outliers
    distribution: Distribution


class CategoricalFeature(CustomBaseModel):
    missing_values: int
    nr_levels: int
    levels: Dict[str, str]
    imbalance: float
    mutual_info: float
    monotonous_filtering: float


class UnstructuredFeature(CustomBaseModel):
    missing_values: int
    vocab_size: int
    relative_vocab: float
    vocab_concentration: float
    entropy: float
    min_vocab: int
    max_vocab: int


class DatetimeFeature(CustomBaseModel):
    pass


class Features(CustomBaseModel):
    numerical_features: Dict[str, NumericalFeature]
    categorical_features: Dict[str, CategoricalFeature]
    unstructured_features: Dict[str, UnstructuredFeature]
    datetime_features: Dict[str, DatetimeFeature]


class Dataset(Document):
    info: Info
    features: Features
    tasks: List[BackLink[Task]] = Field(json_schema_extra={"original_field": "dataset"})

    class Settings:
        name = "datasets"
        keep_nulls = False
        validate_on_save = True
        indexes = [
            IndexModel("info.mlseaUri", name="info.mlseaUri_", unique=True,
                       partialFilterExpression={"info.mlseaUri": {"$exists": True}})
        ]

    class Config:
        arbitrary_types_allowed = True
        populate_by_name = True
        use_enum_values = True
        alias_generator = alias_generator

    def _get_base_meta_dataset_descriptor(self) -> np.ndarray:
        """
        Get the base meta dataset descriptor.

        Returns:
        np.array: The base meta dataset descriptor.
        """
        base_observations = np.log1p(self.info.observations)
        base_features = np.log1p(self.info.nr_total_features)
        base_numerical_ratio = self.info.numerical_ratio
        base_categorical_ratio = self.info.categorical_ratio
        base_datetime_ratio = self.info.datetime_ratio
        base_unstructured_ratio = self.info.unstructured_ratio
        return np.array([
            base_observations,
            base_features,
            base_numerical_ratio,
            base_categorical_ratio,
            base_datetime_ratio,
            base_unstructured_ratio
        ])

    def _get_aggregated_numerical_dataset_descriptor(self) -> np.ndarray:
        """
        Get the aggregated numerical dataset descriptor.

        Returns:
        np.array: The aggregated numerical dataset descriptor.
        """
        numeric_features = self.features.numerical_features
        if numeric_features:
            numeric_stats = []
            for feature in numeric_features.values():
                missing_value_ratio = feature.missing_values / self.info.analyzed_observations
                min_value = np.log1p(abs(feature.min_value))
                max_value = np.log1p(abs(feature.max_value))
                outliers_ratio = feature.outliers.number / self.info.analyzed_observations
                monotonous_filtering = feature.monotonous_filtering
                anova_f1 = np.log1p(feature.anova_f1)
                anova_pvalue = feature.anova_pvalue
                feature_values = [
                    missing_value_ratio,
                    min_value,
                    max_value,
                    outliers_ratio,
                    monotonous_filtering,
                ]
                if any([np.isnan(value) for value in feature_values]):
                    continue
                numeric_stats.append(feature_values)
            numeric_stats = np.array(numeric_stats)
            if numeric_stats.size > 0:
                numeric_agg_mean = np.mean(numeric_stats, axis=0)
                numeric_agg_std = np.std(numeric_stats, axis=0)
            else:
                numeric_agg_mean = np.zeros(7)
                numeric_agg_std = np.zeros(7)
        else:
            numeric_agg_mean = np.zeros(5)
            numeric_agg_std = np.zeros(5)
        return np.hstack([numeric_agg_mean, numeric_agg_std])

    def _get_aggregated_categorical_dataset_descriptor(self) -> np.ndarray:
        """
        Get the aggregated categorical dataset descriptor.

        Returns:
        np.array: The aggregated categorical dataset descriptor.
        """
        categorical_features = self.features.categorical_features
        if categorical_features:
            categorical_stats = []
            for feature in categorical_features.values():
                missing_value_ratio = feature.missing_values / self.info.analyzed_observations
                nr_levels = feature.nr_levels
                imbalance = feature.imbalance
                monotonous_filtering = feature.monotonous_filtering
                mutual_info = feature.mutual_info
                feature_values = [
                    missing_value_ratio,
                    nr_levels,
                    imbalance,
                    monotonous_filtering,
                    mutual_info
                ]
                if any([np.isnan(value) for value in feature_values]):
                    continue
                categorical_stats.append(feature_values)
            categorical_stats = np.array(categorical_stats)
            if categorical_stats.size > 0:
                categorical_agg_mean = np.mean(categorical_stats, axis=0)
                categorical_agg_std = np.std(categorical_stats, axis=0)
            else:
                categorical_agg_mean = np.zeros(5)
                categorical_agg_std = np.zeros(5)
        else:
            categorical_agg_mean = np.zeros(5)
            categorical_agg_std = np.zeros(5)
        return np.hstack([categorical_agg_mean, categorical_agg_std])

    def _get_aggregated_unstructured_dataset_descriptor(self) -> np.ndarray:
        """
        Get the aggregated unstructured dataset descriptor.

        Returns:
        np.array: The aggregated unstructured dataset descriptor.
        """
        unstructured_features = self.features.unstructured_features
        if unstructured_features:
            unstructured_stats = []
            for feature in unstructured_features.values():
                missing_value_ratio = feature.missing_values / self.info.analyzed_observations
                vocab_size = feature.vocab_size
                relative_vocab = feature.relative_vocab
                vocab_concentration = feature.vocab_concentration
                entropy = feature.entropy
                min_vocab = feature.min_vocab
                max_vocab = feature.max_vocab
                feature_values = [
                    missing_value_ratio,
                    vocab_size,
                    relative_vocab,
                    vocab_concentration,
                    entropy,
                    min_vocab,
                    max_vocab
                ]
                if any([np.isnan(value) for value in feature_values]):
                    continue
                unstructured_stats.append(feature_values)
            unstructured_stats = np.array(unstructured_stats)
            if unstructured_stats.size > 0:
                unstructured_agg_mean = np.mean(unstructured_stats, axis=0)
                unstructured_agg_std = np.std(unstructured_stats, axis=0)
            else:
                unstructured_agg_mean = np.zeros(7)
                unstructured_agg_std = np.zeros(7)
        else:
            unstructured_agg_mean = np.zeros(7)
            unstructured_agg_std = np.zeros(7)
        return np.hstack([unstructured_agg_mean, unstructured_agg_std])

    def _get_aggregated_datetime_dataset_descriptor(self) -> np.ndarray:
        """
        Get the aggregated datetime dataset descriptor.

        Returns:
        np.array: The aggregated datetime dataset descriptor.
        """
        datetime_features = self.features.datetime_features
        if datetime_features:
            datetime_stats = []
            for feature in datetime_features.values():
                datetime_stats.append([])  # Placeholder for datetime features, not implemented yet
            datetime_stats = np.array(datetime_stats)
            datetime_agg_mean = np.mean(datetime_stats, axis=0)
            datetime_agg_std = np.std(datetime_stats, axis=0)
        else:
            datetime_agg_mean = np.zeros(0)
            datetime_agg_std = np.zeros(0)
        return np.hstack([datetime_agg_mean, datetime_agg_std])

    def get_dataset_descriptor(self) -> np.ndarray:
        """
        Get the dataset descriptor.

        Returns:
        np.array: The dataset descriptor.
        """
        base_vector = self._get_base_meta_dataset_descriptor()
        numeric_vector = self._get_aggregated_numerical_dataset_descriptor()
        categorical_vector = self._get_aggregated_categorical_dataset_descriptor()
        unstructured_vector = self._get_aggregated_unstructured_dataset_descriptor()
        datetime_vector = self._get_aggregated_datetime_dataset_descriptor()

        final_vector = np.hstack([
            base_vector,
            numeric_vector,
            categorical_vector,
            unstructured_vector,
            datetime_vector
        ])
        return final_vector

    def similarity(self, other: "Dataset", normalizer: DatasetDescriptorNormalizer) -> float:
        """
        Calculate the similarity between two datasets.

        Parameters:
        other (Dataset): The other dataset to compare with.

        Returns:
        float: The similarity between the two datasets.
        """
        self_vector = normalizer.normalize(self.get_dataset_descriptor())
        other_vector = normalizer.normalize(other.get_dataset_descriptor())
        self_norm = np.linalg.norm(self_vector)
        other_norm = np.linalg.norm(other_vector)
        if self_norm == 0 or other_norm == 0:
            return 0.0
        return np.dot(self_vector, other_vector) / (self_norm * other_norm)

#from .task import Task
#Dataset.update_forward_refs()
