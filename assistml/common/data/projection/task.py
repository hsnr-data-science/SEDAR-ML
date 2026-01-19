from beanie import PydanticObjectId, Link
from pydantic import BaseModel, Field

from common.data import Dataset


class EmptyView(BaseModel):
    id: PydanticObjectId = Field(alias="_id")

    class Settings:
        projection = {"id": "$_id"}

class DatasetView(BaseModel):
    id: PydanticObjectId = Field(alias="_id")
    dataset: Link[Dataset]

    class Settings:
        projection = {"id": "$_id"}
