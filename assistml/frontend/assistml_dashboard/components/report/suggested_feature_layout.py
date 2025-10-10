from typing import Union

from dash import html
from dash.dash_table import DataTable

from common.data.dataset import NumericalFeature, CategoricalFeature
from common.dto import DatasetInfoDto


def construct_json_for_feature(feature_name, feature_info: Union[NumericalFeature, CategoricalFeature], missing_values_percent):
    selected_feature_dict = {
        'Name': feature_name,
        'Mutual info': round(feature_info.mutual_info, 3),
        'Missing values percentage': round(missing_values_percent, 3),
        'Monotonous filtering': round(feature_info.monotonous_filtering, 3)
    }
    return selected_feature_dict


def _suggest_features_to_user(dataset_info: DatasetInfoDto, class_feature_type):
    numerical_features_keys = []
    selected_features = {}
    rows_nr = dataset_info.info.observations

    # Numerical features
    selected_features['numerical_features'] = []
    numerical_features = dataset_info.features.numerical_features
    if "numeric" in class_feature_type:
        for feature_name, feature_info in numerical_features.items():
            missing_values_nr = feature_info.missing_values
            missing_values_percent = (missing_values_nr / rows_nr) * 100
            numerical_features_keys = ['Name', 'Missing values percentage', 'Mutual info', 'Monotonous filtering']
            monotonous_filtering = feature_info.monotonous_filtering
            mutual_info = feature_info.mutual_info
            if missing_values_percent < 20 and mutual_info >= 0.01 and 0.5 < monotonous_filtering < 0.9:
                selected_feature_dict = construct_json_for_feature(feature_name, feature_info,
                                                                          missing_values_percent)
                selected_features['numerical_features'].append(selected_feature_dict)
    else:
        for feature_name, feature_info in numerical_features.items():
            missing_values_nr = feature_info.missing_values
            missing_values_percent = (missing_values_nr / rows_nr) * 100
            numerical_features_keys = ['Name', 'Missing values percentage', 'Mutual info', 'Monotonous filtering']
            mutual_info = feature_info.mutual_info
            monotonous_filtering = feature_info.monotonous_filtering
            if missing_values_percent < 20 and mutual_info >= 0.01 and 0.5 < monotonous_filtering < 0.9:
                selected_feature_dict = construct_json_for_feature(feature_name, feature_info,
                                                                          missing_values_percent)
                selected_features['numerical_features'].append(selected_feature_dict)

    # Categorical features
    selected_features['categorical_features'] = []
    categorical_features = dataset_info.features.categorical_features
    for feature_name, feature_info in categorical_features.items():
        missing_values_nr = feature_info.missing_values
        missing_values_percent = (missing_values_nr / rows_nr) * 100
        mutual_info = feature_info.mutual_info
        monotonous_filtering = feature_info.monotonous_filtering
        if missing_values_percent < 20 and mutual_info >= 0.01 and 0.5 < monotonous_filtering < 0.9:
            selected_feature_dict = construct_json_for_feature(feature_name, feature_info,
                                                                          missing_values_percent)
            selected_features['categorical_features'].append(selected_feature_dict)
    return selected_features, numerical_features_keys


def create_suggested_feature_layout(dataset_info: DatasetInfoDto, class_feature_type):
    suggested_features, numerical_features_keys = _suggest_features_to_user(dataset_info, class_feature_type)
    categorical_features_keys = ['Name', 'Missing values percentage', 'Mutual info', 'Monotonous filtering']

    numerical_features_table = html.Div(
        style={'width': '90%'},
        children=[
            html.H5(
                'List of important numerical features',
                style={'font-weight': 'bold'}
            ),
            DataTable(
                data=suggested_features['numerical_features'],
                columns=[{'id': i, 'name': i} for i in numerical_features_keys],

                style_header={
                    'backgroundColor': 'rgb(204, 229, 255)',
                    "text-align": "left", 'justify': 'left',
                    'fontWeight': 'bold',
                },
                style_data_conditional=[
                    {
                        'if': {'row_index': 'odd'},
                        "text-align": "left", 'justify': 'left',
                        'backgroundColor': 'rgb(248, 248, 248)'
                    }
                ],
                style_data={
                    "text-align": "left", 'justify': 'left',
                },
                style_table={
                },
            ),
        ])

    categorical_features_table = html.Div(
        style={'width': '90%'},
        children=[
            html.H5(
                'List of important categorical features',
                style={'font-weight': 'bold'}
            ),
            DataTable(
                data=suggested_features['categorical_features'],
                columns=[{'id': i, 'name': i} for i in categorical_features_keys],

                style_header={
                    'backgroundColor': 'rgb(204, 229, 255)',
                    "text-align": "left", 'justify': 'left',
                    'fontWeight': 'bold',
                },
                style_data_conditional=[
                    {
                        'if': {'row_index': 'odd'},
                        "text-align": "left", 'justify': 'left',
                        'backgroundColor': 'rgb(248, 248, 248)'
                    }
                ],
                style_data={
                    "text-align": "left", 'justify': 'left',
                },
                style_table={},
            ),
        ]
    )
    if suggested_features['numerical_features'] and suggested_features['categorical_features']:
        content = html.Div([numerical_features_table, categorical_features_table], )
    elif not suggested_features['numerical_features']:
        content = html.Div([categorical_features_table], )
    elif not suggested_features['categorical_features']:
        content = html.Div([numerical_features_table], )
    else:
        content = html.Div([])
    return content
