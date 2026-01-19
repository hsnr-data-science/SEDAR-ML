from flash import Flash, Input, Output, State
from quart import g, current_app

from assistml_dashboard.client import BackendClient
from assistml_dashboard.components.report import create_report_layout, create_suggested_feature_layout
from assistml_dashboard.components.sidebar.classifier_preferences_callbacks import register_classifier_preferences_callbacks
from assistml_dashboard.components.sidebar.dataset_characteristics_callbacks import register_dataset_characteristics_callbacks
from common.dto import AnalyseDatasetResponseDto
from common.data.model import Metric
from common.data.task import TaskType


def register_sidebar_callbacks(app: Flash):
    backend: BackendClient = g.backend_client

    register_dataset_characteristics_callbacks(app)
    register_classifier_preferences_callbacks(app)

    @app.callback(
        [
            Output('submit_btn_load_output', 'children'),
            Output('result_section', 'children'),
            Output('report_section', 'children'),
        ],
        [
            Input('submit_button', 'n_clicks'),
        ],
        [
            State('parsed-data', 'data'),
            State('class_label', 'value'),
            State('class_feature_type', 'value'),
            State('feature_type_list', 'value'),
            State('upload-data', 'filename'),
            State('task_type', 'value'),
            State('slider-values-store', 'data'),
        ],
        prevent_initial_call=True
    )
    async def trigger_data_profiler(submit_btn_clicks, serialized_dataframe, class_label, class_feature_type,
                              feature_type_list, csv_filename, task_type, stored_values):
        print(type(submit_btn_clicks))
        response: AnalyseDatasetResponseDto
        response, error = await backend.analyse_dataset(class_label, class_feature_type, feature_type_list)
        if response is None:
            return error, "Feature suggestion not possible", ""

        if response.data_profile is None:
            return response.db_write_status.status, "Feature suggestion not possible", ""

        current_app.logger.debug(f"Dataset_id: {response.db_write_status.dataset_id}")

        suggested_features = create_suggested_feature_layout(response.data_profile, class_feature_type)
        preferences = {Metric(metric): value for metric, value in stored_values.items()}
        report, error = await backend.query(class_feature_type, feature_type_list, preferences, response.db_write_status.dataset_id, csv_filename, TaskType(task_type))

        if report is None:
            return response.db_write_status.status, suggested_features, f"Error while profiling the dataset: {error}"

        report_layout = await create_report_layout(report, error)

        return response.db_write_status.status, suggested_features, report_layout
