from typing import Dict, List, Optional

import pandas as pd
from sklearn.datasets import fetch_openml

from common.data import Dataset
from common.data.dataset import TargetFeatureType, Info
from config import Config
from mlsea import DatasetDto, mlsea_repository as mlsea
from processing.task import process_all as process_all_tasks

from common.data_profiler import DataProfiler, ReadMode
from processing.types import ProcessingOptions


async def process_all_datasets(dataset_ids: List[int] = None, options: ProcessingOptions = ProcessingOptions()):
    count = 0
    offset_id = options.offset.pop('dataset', 0) if options.offset is not None else 0
    while True:
        datasets_df = mlsea.retrieve_datasets_from_openml(dataset_ids, batch_size=10, offset_id=offset_id)
        if datasets_df.empty:
            break

        if options.head is not None:
            datasets_df = datasets_df.head(options.head-count)

        for dataset_dto in datasets_df.itertuples(index=False):
            try:
                dataset_dto = DatasetDto(*dataset_dto)

                print(f"Processing dataset {dataset_dto.openml_dataset_id}")

                dataset: Dataset = await _ensure_dataset_exists(dataset_dto)

                if options.recursive:
                    await process_all_tasks(dataset, options)
            except Exception as e:
                print(f"Error processing dataset {dataset_dto.openml_dataset_id}: {e}")
                with open("error_messages.txt", "a") as f:
                    f.write(f"dataset {dataset_dto.openml_dataset_id}: {e}\n")
                with open("error_datasets.txt", "a") as f:
                    f.write(f"{dataset_dto.openml_dataset_id}\n")
            finally:
                count += 1
                offset_id = dataset_dto.openml_dataset_id

        if options.head is not None and count >= options.head:
            break

async def _ensure_dataset_exists(dataset_dto: DatasetDto):
    dataset: Optional[Dataset] = await Dataset.find_one(
        #Dataset.info.mlsea_uri == dataset_dto.mlsea_dataset_uri
        {"info.mlseaUri": dataset_dto.mlsea_dataset_uri}
    )

    if dataset is not None:
        return dataset

    profiled_dataset = _profile_dataset(dataset_dto.openml_dataset_id,
                                                         dataset_dto.default_target_feature_label)

    dataset = Dataset(**profiled_dataset)
    dataset.info.mlsea_uri = dataset_dto.mlsea_dataset_uri
    await dataset.insert()
    return dataset

def _profile_dataset(openml_dataset_id, default_target_feature_label: str):
    raw_data = fetch_openml(data_id=openml_dataset_id, parser='auto', as_frame=True, cache=Config.OPENML_USE_CACHE)
    df: pd.DataFrame = raw_data['frame']
    details = raw_data['details']

    feature_annotations = [
        'T' if feature_name == default_target_feature_label else
        'N' if feature_type in ['int64', 'float64', 'numeric'] else
        'C' if feature_type in ['category'] else
        'D' if feature_type in ['datetime64'] else
        'U'
        for feature_name, feature_type in df.dtypes.items()
    ]
    feature_annotations_string = '[' + ','.join(feature_annotations) + ']'
    target_feature_type = _recognize_classification_output_type(df, default_target_feature_label)

    data_profiler = DataProfiler(
        dataset_name=details['name'],
        target_label=default_target_feature_label,
        target_feature_type=target_feature_type
    )
    data_info = data_profiler.analyse_dataset(ReadMode.READ_FROM_DATAFRAME, feature_annotations_string,
                                              dataset_df=df)
    return data_info

def _recognize_classification_output_type(df: pd.DataFrame, target_feature: str):
    if df.dtypes[target_feature] == 'category':
        if len(df[target_feature].cat.categories) == 2:
            return TargetFeatureType.BINARY
        elif len(df[target_feature].cat.categories) > 2:
            return TargetFeatureType.CATEGORICAL
        else:
            raise ValueError("Unrecognized type. Category with less than 2 categories")
    elif df.dtypes[target_feature] == 'int64':
        if len(df[target_feature].unique()) == 2:
            return TargetFeatureType.BINARY
        elif len(df[target_feature].unique()) > 2:
            return TargetFeatureType.CATEGORICAL
    elif df.dtypes[target_feature] in ['float64', 'numeric']:
        return TargetFeatureType.NUMERICAL
    else:
        raise ValueError("Unrecognized type")
