from typing import Optional, List, Any, Dict, Union

from beanie import Document, Link
from pydantic import Field, field_serializer, field_validator, confloat, SerializationInfo

from . import Implementation
from .dataset import Dataset
from .implementation import Platform
from .model import Model, Metric
from .task import TaskType
from .utils import CustomBaseModel, alias_generator, encode_dict


class Summary(CustomBaseModel):
    acceptable_models: int
    nearly_acceptable_models: int
    distrust_score: float
    warnings: List[str]


class PerformanceReport(CustomBaseModel):
    quantile_label: str
    normalized_mean: float
    normalized_std: float
    mean: float
    std: float


class PartialHyperparameterConfiguration(CustomBaseModel):
    implementation: Link[Implementation]
    hyperparameters: Dict[str, Any]

    # @field_validator("implementation", mode="before")
    # def validate_implementation(cls, v: Any) -> Link[Implementation]:
    #     if isinstance(v, dict):
    #         try:
    #             return Link[Implementation](DBRef(v["collection"], v["id"]), Implementation)
    #         except ValueError:
    #             raise ValueError("Invalid implementation")
    #     elif not isinstance(v, Link):
    #         raise ValueError("Invalid implementation")
    #     return v

    @field_serializer("implementation")
    def serialize_implementation(self, implementation: Link[Implementation], info: SerializationInfo) -> Union[Link[Implementation], Dict[str, Any]]:
        if info.mode == "json":
            return implementation.to_dict()

        if isinstance(implementation, Link):
            return implementation

        if isinstance(implementation, Implementation):
            return Link[Implementation](implementation.to_ref(), Implementation)

        raise ValueError(f"Unknown implementation type: {type(implementation)}")


class HyperparameterConfigurationReport(CustomBaseModel):
    hyperparameters: List[PartialHyperparameterConfiguration]
    performance: Dict[Metric, PerformanceReport]

    @field_validator("performance", mode="before")
    def validate_preferences(cls, v: Any) -> dict[Metric, PerformanceReport]:
        if not isinstance(v, dict):
            raise ValueError("Metrics must be a dictionary")

        validated: Dict[Metric, PerformanceReport] = {}
        for key, value in v.items():
            if isinstance(key, Metric):
                metric = key
            elif isinstance(key, str):
                try:
                    metric = Metric(key)
                except ValueError:
                    raise KeyError(f"Key {key} is not a valid metric")
            else:
                raise ValueError(f"Key {key} is not a valid metric")
            if not isinstance(value, PerformanceReport):
                try:
                    value = PerformanceReport(**value)
                except ValueError:
                    raise ValueError(f"Value {value} is not a valid PerformanceReport")

            validated[metric] = value
        return validated

    @field_serializer("performance")
    def serialize_preferences(self, performance: dict[Metric, PerformanceReport], info) -> Dict[str, PerformanceReport]:
        return {metric.value: value for metric, value in performance.items()}


class ImplementationDatasetGroupReport(CustomBaseModel):
    dataset: Link[Dataset]
    dataset_name: str
    dataset_similarity: float
    dataset_features: int
    dataset_observations: int
    model_count: int
    configurations: List[HyperparameterConfigurationReport]


class ImplementationGroupReport(CustomBaseModel):
    name: str
    platform: Platform
    overall_score: float
    nr_hparams: int
    nr_dependencies: int
    implementation: Link[Implementation]
    performance: Dict[Metric, PerformanceReport]
    dataset_groups: List[ImplementationDatasetGroupReport]
    default_configuration: Optional[HyperparameterConfigurationReport] = None
    class_name: Optional[str] = None


class Report(CustomBaseModel):
    summary: Summary
    acceptable_models: List[ImplementationGroupReport]
    nearly_acceptable_models: List[ImplementationGroupReport] = Field(list)


class Query(Document):
    made_at: str
    task_type: TaskType
    dataset: Link[Dataset]
    semantic_types: List[str]
    preferences: Dict[Metric, confloat(ge=0, le=1)]
    report: Optional[Report] = None

    class Settings:
        name = "queries"
        keep_nulls = False
        validate_on_save = True
        alias_generator = alias_generator
        bson_encoders = {
            Dict: encode_dict
        }

    @field_validator("preferences", mode="before")
    def validate_preferences(cls, v: Any) -> dict[Metric, Any]:
        return Model.validate_metrics(v)

    @field_validator("task_type", mode="before")
    def convert_task_type(cls, value):
        if isinstance(value, str):
            try:
                return TaskType(value)
            except KeyError:
                raise ValueError(f"Invalid task_type: {value}")
        return value
