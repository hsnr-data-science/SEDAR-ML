import glob
import os
from typing import Any, Optional

import httpx
from pydantic import ValidationError

from common.data.model import Metric
from common.data.task import TaskType
from common.dto import AnalyseDatasetRequestDto, AnalyseDatasetResponseDto, ReportRequestDto, ReportResponseDto


class BackendClient:

    def __init__(self, config: dict):
        self.base_url = config['BACKEND_BASE_URL']
        self.working_dir = config['WORKING_DIR']
        # because processing can take a while, we set a longer timeout
        self.timeout = None if config['DEBUG'] else 12 * 60 * 60 * 1000 # 12 hours

    async def analyse_dataset(self, class_label: str, class_feature_type: str, feature_type_list: str) -> (Optional[AnalyseDatasetResponseDto], Optional[str]):
        url = f"{self.base_url}/analyse-dataset"
        upload_dir = os.path.join(self.working_dir, "uploads")  # TODO: skip saving file on disk
        os.makedirs(upload_dir, exist_ok=True)
        csv_files = glob.glob(os.path.join(upload_dir, "*.csv"))
        arff_files = glob.glob(os.path.join(upload_dir, "*.arff"))
        all_files = sorted(csv_files + arff_files, key=os.path.getmtime)
        file = all_files[-1].split("/")[-1]
        payload = AnalyseDatasetRequestDto(
            class_label=class_label,
            class_feature_type=class_feature_type,
            feature_type_list=feature_type_list
        )
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            with open(os.path.join(upload_dir, file), "rb") as dataset_uploaded:
                file_dict = {
                    "json": (None, payload.model_dump_json(), "application/json"),
                    "file": (str(file), dataset_uploaded, "text/plain")
                }
                response = await client.post(url, files=file_dict)
                if response.status_code != 200:
                    return None, f"Failed to analyse dataset: {response.text}"
                response_json = response.json()
                try:
                    return AnalyseDatasetResponseDto(**response_json), None
                except ValidationError as e:
                    return None, f"Error while parsing response: {e}"

    async def query(
            self,
            class_feature_type,
            feature_type_list,
            preferences: dict[Metric, Any],
            dataset_id: str,
            csv_filename: str,
            task_type: TaskType
    ) -> (Optional[ReportResponseDto], Optional[str]):
        feature_type_list = feature_type_list.replace(' ', '')
        feature_type_list = feature_type_list.replace("'", '')
        feature_type_list = feature_type_list.replace('"', '')
        feature_type_list = list(feature_type_list.strip('[]').split(','))

        url = f"{self.base_url}/query"
        query_dto = ReportRequestDto(
            classification_type=class_feature_type,
            semantic_types=feature_type_list,
            preferences=preferences,
            dataset_id=dataset_id,
            dataset_name=csv_filename,
            task_type=task_type,
        )
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(url=url, json=query_dto.model_dump(by_alias=True),
                                     headers={'Content-Type': 'application/json'})

            if response.status_code != 200:
                return None, response.text

            response_json = response.json()
            try:
                return ReportResponseDto(**response_json), None
            except ValidationError as e:
                return None, f"Error while parsing response: {e}"
