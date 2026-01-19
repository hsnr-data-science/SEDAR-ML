from pydantic import BaseModel

from common.data.dataset import TargetFeatureType


class AnalyseDatasetRequestDto(BaseModel):
    class_label: str
    class_feature_type: TargetFeatureType
    feature_type_list: str
