from typing import NamedTuple


class DatasetDto(NamedTuple):
    mlsea_dataset_uri: str
    openml_dataset_id: int
    openml_dataset_url: str
    title: str
    default_target_feature_label: str


class TaskDto(NamedTuple):
    mlsea_task_uri: str
    openml_task_id: int
    openml_task_url: str
    title: str
    task_type: str
    evaluation_procedure_type: str


class ImplementationDto(NamedTuple):
    mlsea_implementation_uri: str
    openml_flow_id: int
    openml_flow_url: str
    title: str


class SoftwareDto(NamedTuple):
    mlsea_software_uri: str
    software_requirement: str


class RunDto(NamedTuple):
    mlsea_run_uri: str
    openml_run_id: int
    openml_run_url: str
    mlsea_implementation_uri: str


class MetricDto(NamedTuple):
    measure_type: str
    value: str
