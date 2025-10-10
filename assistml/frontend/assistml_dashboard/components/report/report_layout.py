from dash import dcc, html

from assistml_dashboard.components.report.model_report_table_layout import create_implementation_group_report_layout
from common.data.query import Summary
from common.dto import ReportResponseDto


def _generate_summary(summary: Summary):
    distrust = "The distrust score for is: " + str(summary.distrust_score * 100) + "%"
    warnings = summary.warnings
    warnings_string = ""
    no_of_acceptable = summary.acceptable_models
    no_of_nearly_acceptable = summary.nearly_acceptable_models
    if len(warnings) > 0:
        distrust += " and the reason for this is the following:"
        for warning in warnings:
            warnings_string += "\n* " + warning
    else:
        distrust += "."
    return html.Div([
        html.H1('Query results'),
        html.Div([
            html.P(
                "There is " + str(no_of_acceptable) + " acceptable models that match your query and " + str(
                    no_of_nearly_acceptable) + " nearly acceptable models."),
            html.P(distrust),
            dcc.Markdown(warnings_string)
        ])
    ])

async def _generate_table(acceptable_models: [], nearly_acceptable_models):
    return html.Div([
        html.Br(),
        html.H5(children='Acceptable Models', style={'font-weight': 'bold', }),
        html.Div(
            [
                await create_implementation_group_report_layout(implementation_group)
                for implementation_group in acceptable_models
            ],
             style={'display': 'inline-block'}
        ),
        html.Br(),
        html.H5(children='Nearly Acceptable Models', style={'font-weight': 'bold', }),
        html.Div(
            [
                await create_implementation_group_report_layout(implementation_group)
                for implementation_group in nearly_acceptable_models
            ],
            style={'display': 'inline-block'}
        ),
    ],
        style={'width': '90%', 'display': 'inline-block'})

async def create_report_layout(report: ReportResponseDto, response):
    if report is None:
        return html.Div([
                html.H6(
                    children='Execution of Remote R backend terminated with status code ' + str(response.status_code),
                    style={'font-weight': 'bold', }),
            ])

    return html.Div([
        _generate_summary(report.summary),
        await _generate_table(report.acceptable_models, report.nearly_acceptable_models)
    ])
