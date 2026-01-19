from statistics import mean
from typing import Optional, Any, Dict

import openml.runs
from beanie import Link, WriteRules

from common.data import Task, Model, Implementation
from common.data.model import Setup, Parameter, Metric
from common.data.implementation import Platform
from mlsea import mlsea_repository as mlsea
from mlsea.dtos import RunDto
from processing.implementation import find_or_create_implementation
from processing.types import ProcessingOptions

RUN_BASE_URI = "http://w3id.org/mlsea/openml/run/"
EVALUATION_MEASURE_BASE_URI = "http://w3id.org/mlso/vocab/evaluation_measure#"

async def process_all_models(task: Task, options: ProcessingOptions = ProcessingOptions()):
    openml_task_id = int(task.mlsea_uri.split('/')[-1])
    count = 0
    offset_id = options.offset.pop('run', 0) if options.offset is not None else 0
    while True:
        task_runs_df = mlsea.retrieve_all_runs_from_openml_for_task(openml_task_id, batch_size=100, offset_id=offset_id)
        if task_runs_df.empty:
            break

        if options.head is not None:
            task_runs_df = task_runs_df.head(options.head)

        for run_dto in task_runs_df.itertuples(index=False):
            try:
                run_dto = RunDto(*run_dto)

                print(f"Processing run {run_dto.openml_run_id}")

                await _ensure_model_exists(run_dto, task)

            except Exception as e:
                print(f"Error processing run {run_dto.openml_run_id}: {e}")
                with open("error_messages.txt", "a") as f:
                    f.write(f"run {run_dto.openml_run_id}: {e}\n")
                with open("error_runs.txt", "a") as f:
                    f.write(f"{run_dto.openml_run_id}\n")

            finally:
                count += 1
                offset_id = run_dto.openml_run_id

        if options.head is not None and count >= options.head:
            break

async def _ensure_model_exists(run_dto: RunDto, task: Task):
    model: Optional[Model] = await Model.find_one(
        #Model.mlsea_uri == run_dto.mlsea_run_uri
        {  "mlseaUri": run_dto.mlsea_run_uri }
    )

    if model is not None:
        return model

    setup = await _generate_setup(run_dto, task)
    metrics = _generate_metrics(run_dto)

    model = Model(
        mlsea_uri=run_dto.mlsea_run_uri,
        setup=setup,
        metrics=metrics
    )
    await model.insert()
    return model


async def _generate_setup(run_dto: RunDto, task: Task) -> Setup:
    openml_run_id = int(run_dto.openml_run_url.split('/')[-1])
    openml_run = openml.runs.get_run(openml_run_id)
    openml_setup = openml.setups.get_setup(openml_run.setup_id)

    hyper_parameters = []
    if openml_setup.parameters is not None:
        for parameter in openml_setup.parameters.values():
            parameter: openml.setups.OpenMLParameter

            implementation = await find_or_create_implementation(parameter.flow_id)
            if implementation is None:
                raise ValueError(f"Could not find or create implementation for hyper_parameter")

            hyper_parameters.append(Parameter(
                name=parameter.parameter_name,
                data_type=parameter.data_type,
                implementation=Link(implementation.to_ref(), Implementation),
                value=parameter.value,
                default_value=parameter.default_value
            ))
    implementation = await find_or_create_implementation(openml_setup.flow_id)
    if implementation is None:
        raise ValueError(f"Could not find or create implementation for setup")
    if (implementation.platform in [Platform.WEKA]
            and implementation.class_name is None
            and openml_run.setup_string is not None):
        implementation.class_name = openml_run.setup_string.split(' ')[0]
        await implementation.save(link_rule=WriteRules.DO_NOTHING)

    return Setup(
        hyper_parameters=hyper_parameters,
        setup_string=openml_run.setup_string,
        implementation=Link(implementation.to_ref(), Implementation),
        task=Link(task.to_ref(), Task)
    )


def _generate_metrics(run_dto: RunDto) -> Dict[Metric, Any]:
    optional_metric_names = {Metric.TRAINING_TIME}

    # metrics taken from MLSea
    run_metrics_df = mlsea.retrieve_all_metrics_from_openml_for_run(run_dto.openml_run_id)
    metrics = list(set([metric for metric in Metric]) - optional_metric_names)
    run_metrics: Dict[Metric, Any] = {
        metric: run_metrics_df.loc[run_metrics_df.measure_type == f"{EVALUATION_MEASURE_BASE_URI}{metric.value}", 'value'].values[0]
        for metric in metrics
        if f"{EVALUATION_MEASURE_BASE_URI}{metric.value}" in run_metrics_df.measure_type.values
    }

    # metrics taken from OpenML
    openml_run = openml.runs.get_run(run_dto.openml_run_id)
    fold_evaluations = openml_run.fold_evaluations
    if fold_evaluations is not None:
        if "usercpu_time_millis" in fold_evaluations:
            usercpu_time_millis_lists = fold_evaluations['usercpu_time_millis'].values()
            run_metrics[Metric.TRAINING_TIME] = mean([mean(list(x.values())) for x in usercpu_time_millis_lists])

    return run_metrics
