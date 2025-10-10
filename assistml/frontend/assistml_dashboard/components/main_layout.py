from dash import html

from assistml_dashboard.components.header_layout import create_header
from assistml_dashboard.components.sidebar import create_sidebar
from assistml_dashboard.components.content_layout import create_content


async def create_layout():
    header = create_header()
    sidebar = await create_sidebar()
    content = create_content()

    return html.Div(
    [
        html.Div([header]),
        html.Div(
            children=[sidebar]
        ),
        html.Div(
            children=[content]
        )
    ],
)
