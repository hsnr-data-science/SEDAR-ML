from beanie import PydanticObjectId
from pydantic import BaseModel


class EmptyView(BaseModel):
    id: PydanticObjectId

    class Settings:
        projection = {"id": "$_id"}
