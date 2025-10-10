from datetime import datetime
from typing import Any, Dict, Optional

from beanie import Document
from bson import ObjectId
from pymongo import IndexModel

from .model import Metric, Setup
from .utils import alias_generator


class SimilarModels(Document):
    model_id: ObjectId
    query_id: ObjectId
    created_at: datetime
    task_model_idx: int  # index of the model in the task, used for even distribution
    mlsea_uri: Optional[str] = None
    setup: Setup
    metrics: Dict[Metric, Any]

    class Settings:
        name = "similar_models"
        validate_on_save = True
        indexes = [
            IndexModel([("queryId", 1), ("modelId", 1)], name="queryId_modelId_", unique=True),
            IndexModel([("queryId", 1), ("taskModelIdx", 1), ("modelId", 1)], name="queryId_taskModelIdx_modelId_", unique=True),
            IndexModel([("queryId", 1)], name="queryId_"),
            IndexModel("createdAt", name="createdAt_", expireAfterSeconds=12 * 60 * 60),
        ]

    class Config:
        arbitrary_types_allowed = True
        populate_by_name = True
        use_enum_values = True
        alias_generator = alias_generator
