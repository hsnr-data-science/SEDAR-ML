import dash
from flash import Flash, Input, Output, State


def register_content_callbacks(app: Flash):
    @app.callback(
        Output("query_issued_value", "is_open"),
        [Input("query_issued_tag", "n_clicks")],
        [State("query_issued_value", "is_open")],
        prevent_initial_call=True
    )
    async def toggle_accordion(n_clicks, is_open):
        ctx = dash.callback_context
        if not ctx.triggered:
            return False
        else:
            button_id = ctx.triggered[0]["prop_id"].split(".")[0]

        if button_id == "query_issued_tag" and n_clicks:
            print(is_open)
            return not is_open
        else:
            return False