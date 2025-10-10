import base64
import csv
import io
import logging
import os
import traceback

import arff
import pandas as pd
from flash import Flash, Input, Output, State


logger = logging.getLogger(__name__)

def register_dataset_characteristics_callbacks(app: Flash):
    @app.callback(
        [
            Output('output_data_upload', 'children'),
            Output('class_label', 'options'),
            Output('parsed-data', 'data'),
            Output('feature_type_list', 'value')
        ],
        [Input('upload-data', 'contents')],
        [State('upload-data', 'filename')],
        prevent_initial_call=True
    )
    async def update_output(list_of_contents, filename: str):
        content_type, content_string = list_of_contents.split(',')
        decoded = base64.b64decode(content_string)
        upload_dir = os.path.join(app.server.config['WORKING_DIR'], "uploads")
        os.makedirs(upload_dir, exist_ok=True)
        logger.info(f"upload {filename}\n")
        with open(os.path.join(upload_dir, filename), 'w') as csv_file:
            for line in str(decoded.decode('utf-8')).splitlines():
                csv_file.write(line)
                csv_file.write('\n')
        try:
            decoded_content = decoded.decode('utf-8')
            if filename.endswith('.csv'):
                # Assume that the user uploaded a CSV file
                sniffer = csv.Sniffer()
                dialect = sniffer.sniff(decoded_content.splitlines()[0])
                df = pd.read_csv(io.StringIO(decoded_content), delimiter=str(dialect.delimiter))
                feature_list = df.columns
                options = [{'label': feature_name, 'value': feature_name} for feature_name in feature_list]

                logger.info("dtypes before serialization:")
                logger.info(df.dtypes)

                serialized_df = {
                    'data': df.to_json(date_format='iso', orient='split'),
                    'dtypes': df.dtypes.astype(str).to_json(),
                }
                return filename + ' was loaded successfully', options, serialized_df, ''

            elif filename.endswith('.arff'):
                # Assume that the user uploaded an .arff file

                # scipy.io.arff does not support strings nor datetime types
                # data, meta = scipy.io.arff.loadarff(io.StringIO(decoded_content))
                data = arff.load(decoded_content)

                df = pd.DataFrame(data['data'], columns=[x[0] for x in data['attributes']])
                for feature_name, feature_type in data['attributes']:
                    if isinstance(feature_type, list):
                        df[feature_name] = df[feature_name].astype("category")
                feature_list = df.columns
                options = [{'label': feature_name, 'value': feature_name} for feature_name in feature_list]

                attributes = data['attributes']
                feature_types = [
                    'C' if isinstance(feat_type, list) else
                    'N' if feat_type == "NUMERIC" else
                    'D' if feat_type == "DATE" else
                    'U' if feat_type == "STRING" else
                    'T'
                    for feat_name, feat_type in attributes
                ]
                feature_types = "[" + ",".join(feature_types) + "]"
                logger.info(feature_types)

                logger.info("dtypes before serialization:")
                logger.info(df.dtypes)

                serialized_df = {
                    'data': df.to_json(date_format='iso', orient='split'),
                    'dtypes': df.dtypes.astype(str).to_json(),
                }
                return filename + ' was loaded successfully', options, serialized_df, feature_types

            else:
                return 'Invalid input file. Try again by uploading csv or arff file'
        except Exception as e:
            logger.error(e)
            # print stack trace
            traceback.print_exc()
            return 'There was an error processing this file.'