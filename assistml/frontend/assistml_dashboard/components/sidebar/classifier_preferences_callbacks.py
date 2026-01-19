from flash import Flash, Input, Output, State, ALL, MATCH

from assistml_dashboard.components.sidebar.classifier_preferences_layout import get_slider_layout
from common.data.model import Metric


def register_classifier_preferences_callbacks(app: Flash):
    @app.callback(
        [Output("slider-container", "children"),
        Output("slider-values-store", "data")],
        Input("metric-dropdown", "value"),
        State("slider-values-store", "data")
    )
    async def update_sliders(selected_metrics, stored_values):
        if not selected_metrics:
            return [], {}

        stored_values = stored_values or {}

        sliders = []
        for metric in selected_metrics:
            metric = Metric(metric)
            previous_value = stored_values.get(metric.value, 0.45)  # default value
            sliders.extend(get_slider_layout(metric, previous_value))

        return sliders, stored_values

    @app.callback(
        Output({"type": "metric-slider-label", "index": MATCH}, "children"),
        Input({"type": "metric-slider", "index": MATCH}, "value"),
        State({"type": "metric-slider", "index": MATCH}, "id")
    )
    async def display_preferences(value, id):
        metric = Metric(id["index"])
        return f"Selected {metric.display_name}: {value}"

    @app.callback(
        Output("slider-values-store", "data", allow_duplicate=True),
        Input({"type": "metric-slider", "index": ALL}, "value"),
        State("metric-dropdown", "value"),
        prevent_initial_call="initial_duplicate",
    )
    async def store_slider_values(values, selected_metrics):
        if not selected_metrics:
            return {}

        return dict(zip(selected_metrics, values))
