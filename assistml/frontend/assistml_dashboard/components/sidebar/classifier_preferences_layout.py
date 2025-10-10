from dash import html, dcc
import dash_bootstrap_components as dbc
from pydantic import BaseModel, Field

from common.data.model import Metric


class _DefaultClassification(BaseModel):
    type: str = Field(None, alias="_id")
    count: int

def get_slider_layout(metric: Metric, value: float):
    return [
        dbc.Label(f"Select {metric.display_name}",
                  width=7, color="#FFFAF0",
                  style={"text-align": "left", 'justify': 'left', 'font-size': '15px', 'font-weight': 'bold',
                         'width': '100%', "background-color": "transparent", "color": "black"}),
        dcc.Slider(
            id={
                "type": "metric-slider",
                "index": metric.value
            },
            min=0,
            max=1,
            step=0.01,
            value=value,
            marks={
                0: '0',
                0.25: '0.25',
                0.5: '0.5',
                0.75: '0.75',
                1: '1'
            },
            tooltip={"placement": "bottom", "always_visible": True},
        ),
        html.Div(f"Selected {metric.display_name}: {value}", id={
            "type": "metric-slider-label",
            "index": metric.value
        }),
        html.Br(),
    ]

async def create_classifier_preferences():
    default_metric_preferences = {
        Metric.ACCURACY.value: 0.35,
        Metric.PRECISION.value: 0.35,
        Metric.RECALL.value: 0.35,
        Metric.TRAINING_TIME.value: 0.35,
    }
    metric_preferences = html.Div([
        dbc.Label(f"Select what metrics to optimize",
                  width=7, color="#FFFAF0",
                  style={"text-align": "left", 'justify': 'left', 'font-size': '15px', 'font-weight': 'bold',
                         'width': '100%', "background-color": "transparent", "color": "black"}),

        dcc.Dropdown(
            id="metric-dropdown",
            options=[{"label": metric.display_name, "value": metric.value} for metric in Metric],
            multi=True,
            placeholder="Choose one ore more metric...",
            value=list(default_metric_preferences.keys()),
            clearable=True
        ),

        html.Br(),

        dcc.Store(id="slider-values-store", data=default_metric_preferences),

        html.Div(id="slider-container"),
    ])

    return html.Div([
    dbc.Label("Classifier Preferences",
              width=7, color="#FFFAF0",
              style={"text-align": "center", 'justify': 'left', 'font-size': '20px', 'font-weight': 'bold',
                     'width': '100%', "background-color": "transparent", "color": "black"}),
    metric_preferences,
])