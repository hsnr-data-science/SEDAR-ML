from enum import Enum
from typing import Optional, Any, List, Type, Dict, Literal

from beanie import Document, Link
from pydantic import field_validator
from pymongo import IndexModel

from .implementation import Implementation
from .task import Task
from .utils import CustomBaseModel, alias_generator, encode_dict


class Parameter(CustomBaseModel):
    name: str
    data_type: Optional[str] = None
    implementation: Link[Implementation]
    value: Optional[Any] = None
    default_value: Optional[Any] = None


class Metric(Enum):
    AREA_UNDER_CURVE = ("area_under_curve", "Area under curve", "maximize")
    AVERAGE_COST = ("average_cost", "Average cost", "minimize")
    F_MEASURE = ("f_measure", "F-measure", "maximize")
    KAPPA = ("kappa", "Kappa", "maximize")
    KONONENKO_BRANKO_RELATIVE_INFORMATION_SCORE = (
        "kononenko_branko_relative_information_score", "Kononenko Branko relative information score", "maximize"
    )
    MEAN_ABSOLUTE_ERROR = ("mean_absolute_error", "Mean absolute error", "minimize")
    MEAN_PRIOR_ABSOLUTE_ERROR = ("mean_prior_absolute_error", "Mean prior absolute error", "minimize")
    PRECISION = ("precision", "Precision", "maximize")
    ACCURACY = ("accuracy", "Accuracy", "maximize")
    PRIOR_ENTROPY = ("prior_entropy", "Prior entropy", "maximize")
    RECALL = ("recall", "Recall", "maximize")
    RELATIVE_ABSOLUTE_ERROR = ("relative_absolute_error", "Relative absolute error", "minimize")
    ROOT_MEAN_PRIOR_SQUARED_ERROR = ("root_mean_prior_squared_error", "Root mean prior squared error", "minimize")
    ROOT_MEAN_SQUARED_ERROR = ("root_mean_squared_error", "Root mean squared error", "minimize")
    ROOT_RELATIVE_SQUARED_ERROR = ("root_relative_squared_error", "Root relative squared error", "minimize")
    TOTAL_COST = ("total_cost", "Total cost", "minimize")
    TRAINING_TIME = ("training_time", "Training time", "minimize")

    def __new__(cls, value: str, display_name: str, optimization_goal = Literal["minimize", "maximize"], datatype: Type = float):
        self = object.__new__(cls)
        self._value_ = value
        self.display_name = display_name
        self.optimization_goal = optimization_goal
        self.datatype = datatype
        return self

    def __hash__(self):
        return hash(self.value)


class Setup(CustomBaseModel):
    hyper_parameters: List[Parameter]
    setup_string: Optional[str] = None
    implementation: Link[Implementation]
    task: Link[Task] = None

    class Config:
        populate_by_name=True


class Model(Document):
    mlsea_uri: Optional[str] = None
    setup: Setup
    metrics: Dict[Metric, Any]

    class Settings:
        name = "models"
        keep_nulls = False
        validate_on_save = True
        indexes = [
            IndexModel("mlseaUri", name="mlseaUri_", unique=True,
                       partialFilterExpression={"mlseaUri": {"$exists": True}}),
            IndexModel("setup.task.$id", name="setup.task.$id_"),
            IndexModel("setup.task", name="setup.task_"),
        ]
        bson_encoders = {
            Dict: encode_dict
        }

    class Config:
        arbitrary_types_allowed=True,
        populate_by_name = True
        alias_generator = alias_generator

    @field_validator("metrics", mode="before")
    def validate_metrics(cls, v: Any) -> Dict[Metric, Any]:
        if not isinstance(v, dict):
            raise ValueError("Metrics must be a dictionary")

        validated: Dict[Metric, Any] = {}
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

            expected_type = metric.datatype
            if not isinstance(value, expected_type):
                try:
                    value = expected_type(value)
                except Exception:
                    raise ValueError(
                        f"Value {value} for metric {metric.name} must be of type {expected_type}, but is {type(value)}"
                    )

            validated[metric] = value
        return validated
