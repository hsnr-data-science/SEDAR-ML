from dash import html, dcc
import dash_bootstrap_components as dbc

from assistml_dashboard.components.sidebar.dataset_characteristics_layout import create_dataset_characteristics
from assistml_dashboard.components.sidebar.classifier_preferences_layout import create_classifier_preferences
from assistml_dashboard.components.sidebar.task_characteristics_layout import create_task_characteristics


async def create_sidebar():
    dataset_characteristics = create_dataset_characteristics()
    task_characteristics = create_task_characteristics()
    classifier_preferences = await create_classifier_preferences()

    submit_button = html.Div([
        dbc.Button("Analyse Dataset", id="submit_button",
                   color="primary", className="mr-1",
                   style={"justify": "center", 'block': 'True', 'width': '100%', "background-color": "rgb(176,196,222)",
                          "font-color": "black"},
                   ),
        dcc.Loading(
            id="submit_btn_loading",
            type="default",
            children=html.Div(id="submit_btn_load_output", style={"font-weight": "bold", },
                              ),
        )]
    )

    return html.Div(
    children=[
        html.H5(children='Fill in required details and upload dataset', style={'font-weight': 'bold', }),
        html.Br(),
        dataset_characteristics,
        html.Br(),
        task_characteristics,
        html.Br(),
        classifier_preferences,
        submit_button,
    ],
    style={
        "border-radius": "10px",
        "margin": "10px",
        "position": "relative",
        "left": 0,
        "top": 0,
        "bottom": 0,
        "width": '25%',
        "padding": "1rem 1rem",
        "float": "left",
    }
)
