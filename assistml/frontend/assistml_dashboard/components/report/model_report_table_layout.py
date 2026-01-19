from typing import Dict, List, Optional, Union

from beanie import Link
from dash import html
from quart import current_app, g

from common.data.query import ImplementationGroupReport
from common.data import Implementation
from common.data.query import HyperparameterConfigurationReport, ImplementationDatasetGroupReport, \
    PartialHyperparameterConfiguration
from common.utils.document_cache import DocumentCache


def background_color(grade):
    color = 'White'
    if type(grade) == str:
        if grade == 'A+':
            color = 'lightblue'
        elif grade == 'A':
            color = 'green'
        elif grade == 'B':
            color = 'lightgreen'
        elif grade == 'C':
            color = 'gold'
        elif grade == 'D':
            color = 'darkgoldenrod'
        else:
            color = 'red'

    if type(grade) == float:
        if grade >= 0.95:
            color = 'lightblue'
        elif grade >= 0.90:
            color = 'green'
        elif grade >= 0.85:
            color = 'lightgreen'
        elif grade >= 0.75:
            color = 'gold'
        elif grade >= 0.65:
            color = 'darkgoldenrod'
        else:
            color = 'red'

    return color

async def create_recursive_hyperparameter_configuration_tree_layout(
        implementation: Union[Link[Implementation], Implementation],
        hyperparameters: List[PartialHyperparameterConfiguration]
):
    if 'document_cache' not in g:
        current_app.logger.debug("Creating document cache")
        g.document_cache = DocumentCache()
    document_cache: DocumentCache = g.document_cache
    implementation: Implementation = await document_cache.get_implementation(implementation)
    configuration: Optional[PartialHyperparameterConfiguration] = next(
        (param
         for param in hyperparameters
         if param.implementation.to_ref().id == implementation.to_ref().id),
        None
    )
    configuration_values: Optional[Dict] = configuration.hyperparameters if configuration is not None else None
    if configuration is not None:
        hyperparameters.remove(configuration)

    params = []
    # Hyperparameters contained in the implementation
    for key, value in implementation.parameters.items():
        is_implementation = implementation.components is not None and key in implementation.components
        has_config_value = configuration_values is not None and key in configuration_values

        if not has_config_value and not is_implementation and value.default_value is None:
            #pass
            continue
        param_value = configuration_values[key] if has_config_value else value.default_value

        param_classes = ["hyperparameter"]
        if not has_config_value:
            param_classes.append("default")
        if is_implementation:
            param_classes.append("implementation")

        param = [
            html.Span([
                html.Span(key, className="hyperparameter-name"),
                html.Span(param_value, className=f"hyperparameter-value{' undefined' if param_value is None else ''}")
            ], className=" ".join(param_classes)),
        ]
        if is_implementation:
            param.append(
                await create_recursive_hyperparameter_configuration_tree_layout(
                    implementation.components[key], hyperparameters)
            )
        params.append(html.Li(param, className=("undefined" if param_value is None else None)))

    # Hyperparameters not contained in the implementation
    if configuration_values is not None:
        for key, value in configuration_values.items():
            if key in implementation.parameters:
                continue
            param = [
                html.Span([
                    html.Span(key, className="hyperparameter-name"),
                    html.Span(value, className=f"hyperparameter-value{' undefined' if value is None else ''}")
                ], className="hyperparameter additional")
            ]
            params.append(html.Li(param))

    return html.Ul(params)

async def create_hyperparameter_configuration_report_layout(
        main_implementation: Union[Link[Implementation], Implementation],
        configuration_report: HyperparameterConfigurationReport,
        index: int
):
    if 'document_cache' not in g:
        current_app.logger.debug("Creating document cache")
        g.document_cache = DocumentCache()
    document_cache:DocumentCache = g.document_cache

    main_implementation: Implementation = await document_cache.get_implementation(main_implementation)
    partial_hyperparameters_list = configuration_report.hyperparameters.copy()
    root_implementation_trees = [
        html.Li(className="root", children=[
            html.Span(main_implementation.class_name, className="implementation"),
            await create_recursive_hyperparameter_configuration_tree_layout(
                main_implementation, partial_hyperparameters_list
            )
        ]
    )]
    for partial_hyperparameters in partial_hyperparameters_list:
        additional_implementation: Implementation = await document_cache.get_implementation(partial_hyperparameters.implementation)
        root_implementation_trees.append(
            html.Li(className="root additional", children=[
                html.Span(additional_implementation.class_name, className="implementation"),
                await create_recursive_hyperparameter_configuration_tree_layout(
                    additional_implementation, [partial_hyperparameters]
                )
            ])
        )

    return html.Details(className="hyperparameter-configuration-report-container", open=index==0, children=[
        html.Summary(className="header", children=[
            html.Div(className="title", children=f"Hyperparameter Configuration {index+1}"),
        ]),
        html.Div(className="tree-container", children=[
            html.Ul(className="tree", children=root_implementation_trees),
        ]),
    ])

async def create_dataset_group_report_layout(
        group_report: ImplementationDatasetGroupReport,
        implementation: Union[Link[Implementation], Implementation]
):
    return html.Div(className="dataset-group-report-container", children=[
        html.Div(className="header", children=[
            html.Div(className="title", children=[group_report.dataset_name]),
        ]),
        html.Div(className="content-container", children=[
            html.Div(className="sidebar", children=[
                html.Div(className="content-block", children=[
                    html.Label("Dataset Similarity"),
                    html.P(round(group_report.dataset_similarity, 4))
                ]),
                html.Div(className="content-block", children=[
                    html.Label("# Features"),
                    html.P(group_report.dataset_features)
                ]),
                html.Div(className="content-block", children=[
                    html.Label("# Observations"),
                    html.P(group_report.dataset_observations)
                ]),
                html.Div(className="content-block", children=[
                    html.Label("# Models"),
                    html.P(group_report.model_count)
                ]),
            ]),
            html.Div(className="main-content", children=[
                html.Div(className="configurations", children=[
                    html.Div(className="configurations-list", children=[
                        await create_hyperparameter_configuration_report_layout(implementation, configuration, i)
                        for i, configuration in enumerate(group_report.configurations)
                    ])
                ]),
            ]),
        ]),
    ])

async def create_implementation_group_report_layout(group_report: ImplementationGroupReport):
    return html.Div(className="implementation-group-report-container", children=[
        html.Div(className="header", children=[
            html.Div(
                className="score",
                children=[
                    html.Label("Overall Score"),
                    html.P(round(group_report.overall_score, 4))
                ],
                style={"background-color": background_color(group_report.overall_score)},
            ),
            html.Div(className="title-and-metrics-container", children=[
                html.Div(className="title", children=[group_report.name]),
                html.Div(className="metrics", children=[
                    html.Div(className="metric", children=[
                        html.Label(f"{metric_name.display_name}"),
                        html.P(
                            metric_value.quantile_label,
                            style={"background-color": background_color(metric_value.quantile_label)}
                        ),
                        html.P(
                            f"{round(metric_value.mean,2)} ± {round(metric_value.std,2)}",
                            style={"background-color": background_color(metric_value.quantile_label)}
                        )
                    ]) for metric_name, metric_value in group_report.performance.items()
                ])
            ]),
        ]),
        html.Div(className="content-container", children=[
            html.Div(className="sidebar", children=[
                html.Div(className="content-block", children=[
                    html.Label("Platform"),
                    html.P(group_report.platform.display_name)
                ]),
                html.Div(className="content-block", children=[
                    html.Label("# Dependencies"),
                    html.P(group_report.nr_dependencies)
                ]),
            ]),
            html.Div(className="main-content", children=[
                await create_dataset_group_report_layout(dataset_group, group_report.implementation)
                for dataset_group in group_report.dataset_groups
            ]),
        ]),
    ])
