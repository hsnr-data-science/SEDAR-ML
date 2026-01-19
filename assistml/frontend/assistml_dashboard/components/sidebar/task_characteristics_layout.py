from dash import html, dcc
import dash_bootstrap_components as dbc

from common.data.task import TaskType


def create_task_characteristics():
    task_type = html.Div(
        [
            dbc.Label("Select Task Type",
                      width=7, color="#FFFAF0",
                      style={"text-align": "left", 'justify': 'left', 'font-size': '15px', 'font-weight': 'bold',
                             'width': '100%', "background-color": "transparent", "color": "black"}
            ),
            dcc.Dropdown(
                id="task_type",
                options=[
                    {'label': task_type.display_name, 'value': task_type.value} for task_type in TaskType
                ],
                placeholder="Select a Task Type",
                value=TaskType.SUPERVISED_CLASSIFICATION.value,
                style={'width': '100%', 'color': 'black'},
            ),
            html.Br(),
    ])
    return html.Div([
        dbc.Label(
            "Task Characteristics",
            width=7, color="#FFFAF0",
            style={"text-align": "center", 'justify': 'left', 'font-size': '20px', 'font-weight': 'bold',
                   'width': '100%', "background-color": "transparent", "color": "black"}
            ),
        task_type,
    ])
