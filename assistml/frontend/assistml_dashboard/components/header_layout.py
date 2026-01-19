from dash import html


def create_header():
    return html.Div([
    html.H1(
        children='Assist ML')],
    style={
        'textAlign': 'center',
        'color': '#000000',
        'backgroundColor': '#8DAD26'
    }
)
