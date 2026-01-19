from dash import html, dcc
import dash_bootstrap_components as dbc

from common.data.dataset import TargetFeatureType


def create_dataset_characteristics():
    csv_upload = html.Div(
        children=[
            dbc.Label("Upload dataset as a csv file here",
                      width=7, color="#FFFAF0",
                      style={"text-align": "left", 'justify': 'left', 'font-size': '15px', 'font-weight': 'bold',
                             'width': '100%', "background-color": "transparent", "color": "black"}),
            dcc.Upload(
                id='upload-data',
                children=html.Div([
                    'Drag and Drop or ',
                    html.A('Select Files')
                ]),
                style={
                    'width': '100%',
                    'height': '60px',
                    'lineHeight': '60px',
                    'borderWidth': '1px',
                    'borderStyle': 'dashed',
                    'borderRadius': '5px',
                    'textAlign': 'center',
                },
                # Allow multiple files to be uploaded
                multiple=False
            ),
            dcc.Store(id='parsed-data'),
            dcc.Loading(
                id="upload_loading",
                type="default",
                children=html.Div(id="output_data_upload")
            ),
            html.Br(),
        ])

    class_label = html.Div(
        [
            dbc.Label("Select Label of Target Class",
                      width=7, color="#FFFAF0",
                      style={"text-align": "left", 'justify': 'left', 'font-size': '15px', 'font-weight': 'bold',
                             'width': '100%', "background-color": "transparent", "color": "black"}),
            dcc.Dropdown(id="class_label", placeholder='Select target class label',
                         style={'width': '100%', 'color': 'black'},
                         options=[], ),
            html.Br(),
        ])

    class_feature_type = html.Div(
        [
            dbc.Label("Select Datatype of Target Class",
                      width=7, color="#FFFAF0",
                      style={"text-align": "left", 'justify': 'left', 'font-size': '15px', 'font-weight': 'bold',
                             'width': '100%', "background-color": "transparent", "color": "black"}),
            dcc.Dropdown(id="class_feature_type",
                         options=[
                             {'label': feature_type.display_name, 'value': feature_type.value}
                             for feature_type in TargetFeatureType
                         ],
                         placeholder="Select a Datatype",
                         style={'width': '100%', 'color': 'black'}),
            html.Br(),
        ])

    feature_type_list = html.Div(
        [
            dbc.Label("Enter Feature Annotation List",
                      width=7, color="#FFFAF0",
                      style={"text-align": "left", 'justify': 'left', 'font-size': '15px', 'font-weight': 'bold',
                             'width': '100%', "background-color": "transparent", "color": "black"}),
            dbc.Input(id="feature_type_list", placeholder='Enter datatype of all features as a list', value='',
                      type="text",
                      style={'width': '100%', 'color': 'black'}),
            html.Br(),
        ])


    return html.Div([
    dbc.Label("Dataset Characteristics",
              width=7, color="#FFFAF0",
              style={"text-align": "center", 'justify': 'left', 'font-size': '20px', 'font-weight': 'bold',
                     'width': '100%', "background-color": "transparent", "color": "black"}),
    csv_upload,
    class_label,
    class_feature_type,
    feature_type_list,
])
