from __future__ import annotations

from enum import Enum
from typing import Optional, ForwardRef, List, Any, Dict

from beanie import Document, BackLink, Link
from pydantic import Field
from pymongo import IndexModel

from .utils import CustomBaseModel, alias_generator

Task = ForwardRef("Task")

class Platform(Enum):
    SCIKIT_LEARN = "scikit-learn", "Scikit-learn", ["sklearn"]
    WEKA = "weka", "Weka", ["weka", "rweka"]
    TORCH = "torch", "Torch", ["torch"]
    MACHINE_LEARNING_IN_R = "machine-learning-in-r", "Machine Learning in R", ["mlr"]
    MASSIVE_ONLINE_ANALYSIS = "massive-online-analysis", "Massive Online Analysis", ["moa"]
    KERAS = "keras", "Keras", ["keras"]
    RAPIDMINER = "rapidminer", "RapidMiner", ["rapidminer", "rm"]
    MXNET = "mxnet", "MXNet", ["mxnet"]
    TENSORFLOW = "tensorflow", "TensorFlow", ["tensorflow"]
    SHOGUN = "shogun", "Shogun", ["shogun"]
    UNKNOWN = "unknown", "Unknown", None

    def __new__(cls, value: str, display_name: str, short_forms: Optional[List[str]]):
        self = object.__new__(cls)
        self._value_ = value
        self.display_name = display_name
        self.short_forms = short_forms
        return self

    def __hash__(self):
        return hash(self.value)

class Software(CustomBaseModel):
    mlsea_uri: Optional[str] = None
    name: str
    version: str

class Parameter(CustomBaseModel):
    default_value: Optional[Any] = None
    type: Optional[str] = None
    description: Optional[str] = None

class Implementation(Document):
    mlsea_uri: Optional[str] = None
    title: str
    class_name: Optional[str] = None
    dependencies: List[Software]
    platform: Platform
    parameters: Dict[str, Parameter]
    components: Optional[Dict[str, Link[Implementation]]] = None
    description: Optional[str] = None

    class Settings:
        name = "implementations"
        keep_nulls = False
        validate_on_save = True
        indexes = [
            IndexModel("mlseaUri", name="mlseaUri_", unique=True,
                       partialFilterExpression={"mlseaUri": {"$exists": True}})
        ]

    class Config:
        arbitrary_types_allowed = True
        populate_by_name = True
        alias_generator = alias_generator

#from .task import Task
#Implementation.update_forward_refs()
