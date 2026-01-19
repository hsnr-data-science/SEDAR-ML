from dash import html
import dash_bootstrap_components as dbc


def create_content():
    return html.Div(
        children=[
            html.H6(id='report_section', ),
            html.H6(id='result_section', children='Analysis results will get displayed here !!!!',
                    style={'font-weight': 'bold', }),
            html.H6(id='query_issued_tag', ),
            dbc.Collapse(id='query_issued_value', ),
        ],
        style={
            'width': '70%',
            "float": "right",
        },
    )
