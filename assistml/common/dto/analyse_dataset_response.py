from typing import Optional

from common.data.dataset import Info, Features
from common.data.utils import CustomBaseModel


class DatasetInfoDto(CustomBaseModel):
    info: Info
    features: Features

class DbWriteStatusDto(CustomBaseModel):
    status: str
    dataset_id: Optional[str] = None

class AnalyseDatasetResponseDto(CustomBaseModel):
    data_profile: Optional[DatasetInfoDto] = None
    db_write_status: DbWriteStatusDto
