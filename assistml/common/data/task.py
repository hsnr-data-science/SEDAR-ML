from abc import ABC
from enum import Enum
from typing import List, ForwardRef, Optional

from beanie import Document, Link, BackLink
from pymongo import IndexModel

from .dataset import Dataset
from .implementation import Implementation
from .utils import alias_generator

Model = ForwardRef("Model")


class TaskType(Enum):
    SUPERVISED_CLASSIFICATION = "Supervised_Classification", "Supervised Classification"
    SUPERVISED_REGRESSION = "Supervised_Regression", "Supervised Regression"
    LEARNING_CURVE = "Learning_Curve", "Learning Curve"
    SUPERVISED_DATASTREAM_CLASSIFICATION = "Supervised_Datastream_Classification", "Supervised Datastream Classification"
    CLUSTERING = "Clustering", "Clustering"
    MACHINE_LEARNING_CHALLENGE = "Machine_Learning_Challenge", "Machine Learning Challenge"
    SURVIVAL_ANALYSIS = "Survival_Analysis", "Survival Analysis"
    SUBGROUP_DISCOVERY = "Subgroup_Discovery", "Subgroup Discovery"
    MULTITASK_REGRESSION = "Multitask_Regression", "Multitask Regression"

    def __new__(cls, value: str, display_name: str):
        self = object.__new__(cls)
        self._value_ = value
        self.display_name = display_name
        return self

    def __hash__(self):
        return hash(self.value)

class Task(Document):
    mlsea_uri: Optional[str] = None
    task_type: TaskType
    dataset: Link[Dataset]
    related_implementations: Optional[List[Link[Implementation]]] = None
    #models: Optional[List[BackLink[Model]]] = Field(None, json_schema_extra={"original_field": "setup.task"})  # nested backlinks seem not to be supported by beanie

    class Settings:
        name = "tasks"
        keep_nulls = False
        validate_on_save = True
        use_enum_values = True
        is_root = True
        indexes = [
            IndexModel("mlseaUri", name="mlseaUri_", unique=True,
                       partialFilterExpression={"mlseaUri": {"$exists": True}}),
            IndexModel("taskType", name="taskType_"),
        ]

    class Config:
        arbitrary_types_allowed = True
        populate_by_name = True
        use_enum_values = True
        alias_generator = alias_generator


class SupervisedTask(Task, ABC):
    target_name: str

class ClassificationTask(SupervisedTask):
    class_labels: Optional[List[str]] = None

class RegressionTask(SupervisedTask):
    pass

class ClusteringTask(Task):
    target_name: Optional[str] = None

class LearningCurveTask(ClassificationTask):
    pass

#from .model import Model
#Task.model_rebuild()
