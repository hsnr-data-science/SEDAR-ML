from enum import Enum
from string import Template

from . import sparql_queries as sq


class Query(Enum):
    """
    Enum class to define SPARQL queries and associated cache filename prefixes.

    Attributes:
        RETRIEVE_ALL_DATASETS_FROM_OPENML (Template, str): Query to retrieve all datasets from OpenML.
        RETRIEVE_DATASET_FROM_OPENML (Template, str): Query to retrieve a specific dataset from OpenML.
        RETRIEVE_ALL_TASKS_FROM_OPENML_FOR_DATASET (Template, str): Query to retrieve all tasks from OpenML for a specific dataset.
        RETRIEVE_ALL_EVALUATION_PROCEDURE_TYPES_FROM_OPENML_FOR_TASK (Template, str): Query to retrieve all evaluation procedure types from OpenML for a specific task.

    Args:
        query (Template): The SPARQL query as a template.
        cached_filename_prefix (str): The prefix for the cache filename.
    """
    RETRIEVE_TASK_ID_FOR_RUN_ID = sq.RETRIEVE_TASK_ID_FOR_RUN_ID, "openml_run_task_id"
    RETRIEVE_DATASET_ID_FOR_TASK_ID = sq.RETRIEVE_DATASET_ID_FOR_TASK_ID, "openml_task_dataset_id"
    RETRIEVE_ALL_DATASETS_FROM_OPENML = sq.RETRIEVE_ALL_DATASETS_FROM_OPENML, "openml_dataset_all"
    RETRIEVE_BATCHED_DATASETS_FROM_OPENML = sq.RETRIEVE_BATCHED_DATASETS_FROM_OPENML, "openml_dataset_batched"
    RETRIEVE_DATASET_FROM_OPENML = sq.RETRIEVE_DATASET_FROM_OPENML, "openml_dataset"
    RETRIEVE_DATASETS_FROM_OPENML = sq.RETRIEVE_DATASETS_FROM_OPENML, "openml_datasets"
    RETRIEVE_ALL_TASKS_FROM_OPENML_FOR_DATASET = sq.RETRIEVE_ALL_TASKS_FROM_OPENML_FOR_DATASET, "openml_dataset_tasks"
    RETRIEVE_BATCHED_TASKS_FROM_OPENML_FOR_DATASET = sq.RETRIEVE_BATCHED_TASKS_FROM_OPENML_FOR_DATASET, "openml_dataset_tasks_batched"
    RETRIEVE_ALL_TASKS_WITH_TYPE_FROM_OPENML_FOR_DATASET = sq.RETRIEVE_ALL_TASKS_WITH_TYPE_FROM_OPENML_FOR_DATASET, "openml_dataset_tasks_with_type"
    RETRIEVE_BATCHED_TASKS_WITH_TYPE_FROM_OPENML_FOR_DATASET = sq.RETRIEVE_BATCHED_TASKS_WITH_TYPE_FROM_OPENML_FOR_DATASET, "openml_dataset_tasks_with_type_batched"
    RETRIEVE_ALL_EVALUATION_PROCEDURE_TYPES_FROM_OPENML_FOR_TASK = sq.RETRIEVE_ALL_EVALUATION_PROCEDURE_TYPES_FROM_OPENML_FOR_TASK, "openml_task_eval_procedure_types"
    RETRIEVE_ALL_IMPLEMENTATIONS_FROM_OPENML_FOR_TASK = sq.RETRIEVE_ALL_IMPLEMENTATIONS_FROM_OPENML_FOR_TASK, "openml_task_implementations"
    RETRIEVE_BATCHED_IMPLEMENTATIONS_FROM_OPENML_FOR_TASK = sq.RETRIEVE_BATCHED_IMPLEMENTATIONS_FROM_OPENML_FOR_TASK, "openml_task_implementations_batched"
    RETRIEVE_IMPLEMENTATION_FROM_OPENML = sq.RETRIEVE_IMPLEMENTATION_FROM_OPENML, "openml_implementation"
    RETRIEVE_ALL_DEPENDENCIES_FROM_OPENML_FOR_IMPLEMENTATION = sq.RETRIEVE_ALL_DEPENDENCIES_FROM_OPENML_FOR_IMPLEMENTATION, "openml_implementation_dependencies"
    RETRIEVE_ALL_RUNS_FROM_OPENML_FOR_TASK = sq.RETRIEVE_ALL_RUNS_FROM_OPENML_FOR_TASK, "openml_task_runs"
    RETRIEVE_BATCHED_RUNS_FROM_OPENML_FOR_TASK = sq.RETRIEVE_BATCHED_RUNS_FROM_OPENML_FOR_TASK, "openml_task_runs_batched"
    RETRIEVE_ALL_METRICS_FROM_OPENML_FOR_RUN = sq.RETRIEVE_ALL_METRICS_FROM_OPENML_FOR_RUN, "openml_run_metrics"

    def __init__(self, query: Template, cached_filename_prefix: str):
        self.query = query
        self.cached_filename_prefix = cached_filename_prefix

    def __call__(self, **kwargs):
        """
        Executes the query with the given parameters.

        Args:
            **kwargs: Keyword arguments for template substitution.

        Returns:
            str: The substituted query.
        """
        return self.query.substitute(PREFIXES=sq.PREFIXES, **kwargs)
