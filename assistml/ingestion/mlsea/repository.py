import os
import time
from typing import List, Optional

import pandas as pd
import sparql_dataframe

from config import Config
from common.data.task import TaskType
from .query import Query


class MLSeaRepository:
    """
    Class to interact with the MLSea SPARQL endpoint and retrieve data.

    Args:
        sparql_endpoint (str): The URL of the SPARQL endpoint.
        use_cache (bool): Indicates whether to use cached results. Handful for development.
        cache_dir_path (str): The path to the directory where to store the cached results.
    """

    def __init__(self,
            sparql_endpoint: str = Config.MLSEA_SPARQL_ENDPOINT,
            use_cache: bool = Config.MLSEA_USE_CACHE,
            cache_dir_path: str = Config.MLSEA_CACHE_DIR,
            retries: int = 10,
            rate_limit: int = Config.MLSEA_RATE_LIMIT
    ):
        self._sparql_endpoint = sparql_endpoint
        self._use_cache = use_cache
        self._cache_dir_path = cache_dir_path
        self._retries = retries
        self._rate_limit = rate_limit  # maximum number of requests per minute
        self._last_request_time = 0

    def retrieve_task_id_for_run_id(self, run_id: int):
        """
        Retrieves the task ID for a specific run.

        Args:
            run_id (int): The ID of the run.

        Returns:
            int: The ID of the task.
        """
        return int(self._execute_query_with_retries(Query.RETRIEVE_TASK_ID_FOR_RUN_ID, runId=run_id)["task_id"].iloc[0])

    def retrieve_dataset_id_for_task_id(self, task_id: int):
        """
        Retrieves the dataset ID for a specific task.

        Args:
            task_id (int): The ID of the task.

        Returns:
            int: The ID of the dataset.
        """
        return int(self._execute_query_with_retries(Query.RETRIEVE_DATASET_ID_FOR_TASK_ID, taskId=task_id)["dataset_id"].iloc[0])

    def retrieve_datasets_from_openml(self, dataset_ids: List[int] = None, batch_size: int = 0, offset_id: int = 0):
        """
        Retrieves specific datasets from OpenML or all datasets if no ID is provided.

        Args:
            dataset_ids (List[int], optional): The IDs of the datasets to retrieve. Defaults to None.
            batch_size (int, optional): The number of datasets to retrieve in a batch. Defaults to 0, which means all datasets.
            offset_id (int, optional): The offset from which to start retrieving datasets (exclusively). Defaults to 0.

        Returns:
            pd.DataFrame: The DataFrame with the retrieved datasets.
        """
        if dataset_ids is None:
            if batch_size > 0:
                return self._execute_query_with_retries(Query.RETRIEVE_BATCHED_DATASETS_FROM_OPENML, limit=batch_size, offsetId=offset_id)
            return self._execute_query_with_retries(Query.RETRIEVE_ALL_DATASETS_FROM_OPENML)
        dataset_ids = " ".join([f"mlsea_openml_dataset:{dataset_id}" for dataset_id in dataset_ids])
        return self._execute_query_with_retries(Query.RETRIEVE_DATASETS_FROM_OPENML, datasetId=dataset_ids)

    def retrieve_all_tasks_from_openml_for_dataset(self, dataset_id: int, batch_size: int = 0, offset_id: int = 0, task_type: Optional[TaskType] = None):
        """
        Retrieves all tasks from OpenML for a specific dataset.

        Args:
            dataset_id (int): The ID of the dataset.
            batch_size (int, optional): The number of tasks to retrieve in a batch. Defaults to 0, which means all tasks.
            offset_id (int, optional): The offset from which to start retrieving tasks (exclusively). Defaults to 0.
            task_type (TaskType, optional): The type of the tasks to retrieve. Defaults to None, which means all tasks.

        Returns:
            pd.DataFrame: The DataFrame with the retrieved tasks.
        """
        if task_type is not None:
            task_type_concept = task_type.value
            # MLSO-TT is not consistent
            if task_type in [TaskType.SURVIVAL_ANALYSIS, TaskType.SUBGROUP_DISCOVERY, TaskType.MACHINE_LEARNING_CHALLENGE]:
                task_type_concept = task_type_concept.replace("_", "")
            if task_type == TaskType.LEARNING_CURVE:
                task_type_concept += "_Estimation"

            if batch_size > 0:
                return self._execute_query_with_retries(Query.RETRIEVE_BATCHED_TASKS_WITH_TYPE_FROM_OPENML_FOR_DATASET, datasetId=dataset_id, taskTypeConcept=task_type_concept, limit=batch_size, offsetId=offset_id)
            return self._execute_query_with_retries(Query.RETRIEVE_ALL_TASKS_WITH_TYPE_FROM_OPENML_FOR_DATASET, datasetId=dataset_id, taskTypeConcept=task_type_concept)

        if batch_size > 0:
            return self._execute_query_with_retries(Query.RETRIEVE_BATCHED_TASKS_FROM_OPENML_FOR_DATASET, datasetId=dataset_id, limit=batch_size, offsetId=offset_id)
        return self._execute_query_with_retries(Query.RETRIEVE_ALL_TASKS_FROM_OPENML_FOR_DATASET, datasetId=dataset_id)

    def retrieve_all_evaluation_procedure_types_from_openml_for_task(self, task_id: int):
        """
        Retrieves all evaluation procedure types from OpenML for a specific task.

        Args:
            task_id (int): The ID of the task.

        Returns:
            pd.DataFrame: The DataFrame with the retrieved evaluation procedure types.
        """
        return self._execute_query_with_retries(Query.RETRIEVE_ALL_EVALUATION_PROCEDURE_TYPES_FROM_OPENML_FOR_TASK,
                                                taskId=task_id)

    def retrieve_all_implementations_from_openml_for_task(self, task_id: int, batch_size: int = 0, offset_id: int = 0):
        """
        Retrieves all implementations from OpenML for a specific task.

        Args:
            task_id (int): The ID of the task.
            batch_size (int, optional): The number of implementations to retrieve in a batch. Defaults to 0, which means all implementations.
            offset_id (int, optional): The offset from which to start retrieving implementations (exclusively). Defaults to 0.

        Returns:
            pd.DataFrame: The DataFrame with the retrieved implementations.
        """
        if batch_size > 0:
            return self._execute_query_with_retries(Query.RETRIEVE_BATCHED_IMPLEMENTATIONS_FROM_OPENML_FOR_TASK, taskId=task_id, limit=batch_size, offsetId=offset_id)
        return self._execute_query_with_retries(Query.RETRIEVE_ALL_IMPLEMENTATIONS_FROM_OPENML_FOR_TASK, taskId=task_id)

    def retrieve_implementation_from_openml(self, implementation_id: int):
        """
        Retrieves a specific implementation from OpenML.

        Args:
            implementation_id (int): The ID of the implementation.

        Returns:
            pd.DataFrame: The DataFrame with the retrieved implementation.
        """
        return self._execute_query_with_retries(Query.RETRIEVE_IMPLEMENTATION_FROM_OPENML, implementationId=implementation_id)

    def retrieve_dependencies_from_openml_for_implementation(self, implementation_id: int):
        """
        Retrieves all dependencies from OpenML for a specific implementation.

        Args:
            implementation_id (int): The ID of the implementation.

        Returns:
            pd.DataFrame: The DataFrame with the retrieved dependencies.
        """
        return self._execute_query_with_retries(Query.RETRIEVE_ALL_DEPENDENCIES_FROM_OPENML_FOR_IMPLEMENTATION,
                                                implementationId=implementation_id)

    def retrieve_all_runs_from_openml_for_task(self, task_id: int, batch_size: int = 0, offset_id: int = 0):
        """
        Retrieves all runs from OpenML for a specific task.

        Args:
            task_id (int): The ID of the task.
            batch_size (int, optional): The number of runs to retrieve in a batch. Defaults to 0, which means all runs.
            offset_id (int, optional): The offset from which to start retrieving runs (exclusively). Defaults to 0.

        Returns:
            pd.DataFrame: The DataFrame with the retrieved runs.
        """
        if batch_size > 0:
            return self._execute_query_with_retries(Query.RETRIEVE_BATCHED_RUNS_FROM_OPENML_FOR_TASK, taskId=task_id, limit=batch_size, offsetId=offset_id)
        return self._execute_query_with_retries(Query.RETRIEVE_ALL_RUNS_FROM_OPENML_FOR_TASK, taskId=task_id)

    def retrieve_all_metrics_from_openml_for_run(self, run_id: int):
        """
        Retrieves all metrics from OpenML for a specific run.

        Args:
            run_id (int): The ID of the run.

        Returns:
            pd.DataFrame: The DataFrame with the retrieved metrics.
        """
        return self._execute_query_with_retries(Query.RETRIEVE_ALL_METRICS_FROM_OPENML_FOR_RUN, runId=run_id)

    def _execute_query(self, query: Query, **params):
        """
        Executes the given query and returns the results as a DataFrame. Uses cache if enabled.

        Args:
            query (Query): The query to execute.
            **params: Parameters for the query.

        Returns:
            pd.DataFrame: The DataFrame with the query results.
        """
        file_path = ''
        if self._use_cache:
            params_str = "_".join([str(value) for value in params.values()])
            filename = f"{query.cached_filename_prefix}_{params_str}.csv" if params else f"{query.cached_filename_prefix}.csv"
            file_path = os.path.join(self._cache_dir_path, filename)
            os.makedirs(self._cache_dir_path, exist_ok=True)
            try:
                return pd.read_csv(file_path)
            except FileNotFoundError:
                #print("No cached response found for query, querying endpoint...")
                pass

        self._ensure_rate_limit()
        result_df = sparql_dataframe.get(self._sparql_endpoint, query(**params))
        if self._use_cache:
            result_df.to_csv(file_path, index=False)

        return result_df

    def _execute_query_with_retries(self, query: Query, **params):
        """
        Executes the given query with retries and returns the results as a DataFrame.

        Args:
            query (Query): The query to execute.
            **params: Parameters for the query.

        Returns:
            pd.DataFrame: The DataFrame with the query results.
        """
        backoff_time = 1  # start with 1 second
        for try_no in range(self._retries):
            try:
                return self._execute_query(query, **params)
            except Exception as ex:
                if try_no == self._retries - 1:
                    raise ex
                print(f"Error executing query: {ex}")
                print(f"Retrying in {backoff_time} seconds...")
                time.sleep(backoff_time)
                print(f"Retrying... ({try_no + 1})")
                backoff_time *= 2

    def _ensure_rate_limit(self):
        """
        Ensures the rate limit is not exceeded.
        """
        current_time = time.time()
        elapsed_time = current_time - self._last_request_time
        if elapsed_time < 60 / self._rate_limit:
            time.sleep(60 / self._rate_limit - elapsed_time)
        self._last_request_time = time.time()

mlsea_repository = MLSeaRepository()
