from datetime import datetime
from typing import Any

from beanie import Document
from bson import ObjectId
from pymongo import ASCENDING, IndexModel

from .utils import alias_generator


class DatasetSimilarity(Document):
    dataset_id: ObjectId
    query_id: ObjectId
    created_at: datetime
    total_features: int
    total_matches: int
    matching_numerical: [Any]
    matching_categorical: [Any]
    similarity3: float
    has_sim_1: bool
    has_sim_2: bool
    has_sim_3: bool

    class Settings:
        name = "dataset_similarities"
        validate_on_save = True
        indexes = [
            IndexModel([("datasetId", ASCENDING), ("queryId", ASCENDING)], name="datasetId_queryId_", unique=True),
            IndexModel([("queryId", ASCENDING), ("datasetId", ASCENDING)], name="queryId_datasetId_", unique=True),
            IndexModel("queryId", name="queryId_"),
            IndexModel("createdAt", name="createdAt_", expireAfterSeconds=12*60*60),
        ]

    class Config:
        arbitrary_types_allowed = True
        populate_by_name = True
        use_enum_values = True
        alias_generator = alias_generator
