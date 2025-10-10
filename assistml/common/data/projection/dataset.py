from beanie import PydanticObjectId

from common.data.dataset import Info, Features
from common.data.utils import CustomBaseModel


class EmptyView(CustomBaseModel):
    id: PydanticObjectId

    class Settings:
        projection = {"id": "$_id"}


class InfoView(CustomBaseModel):
    id: PydanticObjectId
    info: Info

    class Settings:
        projection = {"id": "$_id", "info": 1}


class DatasetNameAndFeaturesView(CustomBaseModel):
    id: PydanticObjectId
    dataset_name: str
    features: Features

    class Settings:
        projection = {"id": "$_id", "dataset_name": "$info.datasetName", "features": 1}
