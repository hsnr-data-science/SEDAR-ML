import re
from typing import Dict, Optional

import openml.tasks

from common.data import Dataset, Task
from common.data.task import TaskType, ClassificationTask, RegressionTask, ClusteringTask, LearningCurveTask
from mlsea import mlsea_repository as mlsea
from mlsea.dtos import TaskDto
from processing.model import process_all_models
from processing.implementation import process_all_implementations
from processing.types import ProcessingOptions

MLSO_TT_BASE_URI = "http://w3id.org/mlso/vocab/ml_task_type#"

async def process_all(dataset: Dataset, options: ProcessingOptions = ProcessingOptions()):
    dataset_id = int(dataset.info.mlsea_uri.split('/')[-1])
    count = 0
    offset_id = options.offset.pop('task', 0) if options.offset is not None else 0
    while True:
        dataset_tasks_df = mlsea.retrieve_all_tasks_from_openml_for_dataset(dataset_id, batch_size=100, offset_id=offset_id, task_type=options.task_type)
        if dataset_tasks_df.empty:
            break

        if options.head is not None:
            dataset_tasks_df = dataset_tasks_df.head(options.head-count)

        for task_dto in dataset_tasks_df.itertuples(index=False):
            try:
                task_dto = TaskDto(*task_dto)

                print(f"Processing task {task_dto.openml_task_id}")

                task: Task = await _ensure_task_exists(task_dto, dataset)

                if options.recursive:
                    await process_all_implementations(task, options)
                    await process_all_models(task, options)

            except Exception as e:
                print(f"Error processing task {task_dto.openml_task_id}: {e}")
                with open("error_messages.txt", "a") as f:
                    f.write(f"task {task_dto.openml_task_id}: {e}\n")
                with open("error_tasks.txt", "a") as f:
                    f.write(f"{task_dto.openml_task_id}\n")

            finally:
                count += 1
                offset_id = task_dto.openml_task_id

        if options.head is not None and count >= options.head:
            break


async def _ensure_task_exists(task_dto: TaskDto, dataset: Dataset):
    task: Optional[Task] = await Task.find_one(
        #Task.mlsea_uri == task_dto.mlsea_task_uri,
        {"mlseaUri": task_dto.mlsea_task_uri},
        with_children=True
    )

    if task is not None:
        return task

    task = _parse_task(task_dto, dataset)

    await task.insert()
    return task

def _parse_task_type(task_type_concept: str) -> TaskType:
    if not task_type_concept.startswith(MLSO_TT_BASE_URI):
        raise ValueError(f"Task type concept {task_type_concept} does not start with {MLSO_TT_BASE_URI}")

    task_type_string = task_type_concept[len(MLSO_TT_BASE_URI):]

    # MLSO-TT is not consistently in Capitalized_Snake_Case
    task_type_string = re.sub(r'(?<!^)(?<!_)([A-Z])', r'_\1', task_type_string)

    # MLSO-TT is not consistent with following OpenML task types
    if task_type_string == "Learning_Curve_Estimation":
        task_type_string = "Learning_Curve"

    return TaskType(task_type_string)

def _parse_task(task_dto, dataset) -> Task:
    openml_task: openml.tasks.OpenMLTask = openml.tasks.get_task(task_dto.openml_task_id)

    task_type = _parse_task_type(task_dto.task_type)
    task: Task
    if task_type == TaskType.SUPERVISED_CLASSIFICATION:
        openml_task: openml.tasks.OpenMLClassificationTask
        task = ClassificationTask(
            task_type=task_type,
            dataset=dataset,
            mlsea_uri=task_dto.mlsea_task_uri,
            target_name=openml_task.target_name,
            class_labels=openml_task.class_labels
        )
    elif task_type == TaskType.SUPERVISED_REGRESSION:
        openml_task: openml.tasks.OpenMLRegressionTask
        task = RegressionTask(
            task_type=task_type,
            dataset=dataset,
            mlsea_uri=task_dto.mlsea_task_uri,
            target_name=openml_task.target_name
        )
    elif task_type == TaskType.CLUSTERING:
        openml_task: openml.tasks.OpenMLClusteringTask
        task = ClusteringTask(
            task_type=task_type,
            dataset=dataset,
            mlsea_uri=task_dto.mlsea_task_uri,
            target_name=openml_task.target_name
        )
    elif task_type == TaskType.LEARNING_CURVE:
        openml_task: openml.tasks.OpenMLLearningCurveTask
        task = LearningCurveTask(
            task_type=task_type,
            dataset=dataset,
            mlsea_uri=task_dto.mlsea_task_uri,
            target_name=openml_task.target_name,
            class_labels=openml_task.class_labels
        )
    else:
        task = Task(
            task_type=task_type,
            dataset=dataset,
            mlsea_uri=task_dto.mlsea_task_uri
        )

    return task
