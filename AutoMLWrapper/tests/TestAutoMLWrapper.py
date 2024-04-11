import unittest
from parameterized import parameterized
import pandas as pd
from automlwrapper.AutoMLWrapper import AutoMLWrapper
from DataSamples import glass_df, mnist_byte_df, m4_df, mnist_tp
from automlwrapper.SedarDataLoader import SedarDataLoader
from sedarapi import SedarAPI


hp = {'epochs':20, 'time_limit': 60*5}#, 'num_trials': 4,}
class TestAutoMLWrapper(unittest.TestCase):

    # =================================================================================================================================#
    # ---------------------------------------------------------------------------------------------------------------------------------#
    # =================================================================================================================================#
    @parameterized.expand([
        ("autosklearn", "tabular", "classification", "multiclass", glass_df, 'Type'),

        ("autokeras", "tabular", "classification", "multiclass", glass_df, 'Type'),
        ("autokeras", "image", "classification", "multiclass", mnist_tp, 'label'),
        ("autokeras", "timeseries", "forecast", "forecast", m4_df, 'target'),
        
        ("autogluon", "tabular", "classification", "multiclass", glass_df, 'Type'),
        ("autogluon", "image", "classification", "multiclass", mnist_byte_df, 'label'),
        ("autogluon", "timeseries", "forecast", "forecast", m4_df, 'target'),

    ])
    def test_auto_ml_wrapper(self, library, data_type, task_type, problem_type, data_sample, target_column):

        with self.subTest(library=library,
                          data_type=data_type,
                          task_type=task_type,
                          problem_type=problem_type,
                          data_sample=data_sample,
                          target_column=target_column
                          ):

            wrapper = AutoMLWrapper(library)

            wrapper.Initialize(data_sample, target_column, task_type, data_type, problem_type)
            wrapper.Train(data_sample, target_column, task_type, data_type, problem_type, hp)

            wrapper.Evaluate(data_sample, target_column)
    
    # =================================================================================================================================#
    # ---------------------------------------------------------------------------------------------------------------------------------#
    # =================================================================================================================================#
    @parameterized.expand([
        #("workspace", "dataset", "file_save_location", "unzip_location", TYPE),
        ("13b4787c3e454649aa05a4cd680edc37", "986f2e837ca44f3e8c0ee7d2dc0c4287",
          "./data/sedar_raw/zip", "./data/sedar_raw/unzip", "STRUCTURED"),

        ("13b4787c3e454649aa05a4cd680edc37", "b5b74391e41e4634a54d5cffa059663b",
          "./data/sedar_raw/zip", "./data/sedar_raw/unzip", "IM_OBJECT"),

        ("13b4787c3e454649aa05a4cd680edc37", "324ea420125d4167a76151b62368c4ad",
          "./data/sedar_raw/zip", "./data/sedar_raw/unzip", "IM_SEGMENT"),

        ("13b4787c3e454649aa05a4cd680edc37", "513c6b1ee46b478c8e0925a098d2f387",
          "./data/sedar_raw/zip", "./data/sedar_raw/unzip", "IM_CLASS"),
    ])
    def test_sedar_data_loader(self, workspace, dataset, file_save_location, unzip_location, type):
        base_url = "http://192.168.220.107:5000"
        email = "admin"
        password = "admin"

        sedar = SedarAPI(base_url)
        sedar.connection.logger.setLevel("INFO")
        sedar.login(email, password)


        with self.subTest(workspace=workspace, dataset=dataset, file_save_location=file_save_location, unzip_location=unzip_location):
            DataLoader = SedarDataLoader(sedar)
            r = DataLoader.query_data(workspace, dataset, query=None, file_save_location=file_save_location)

            if type == "STRUCTURED":
                if not isinstance(r, pd.DataFrame):
                    raise Exception("Data is not a pandas DataFrame")
                
            elif type == "IM_OBJECT":
                df = DataLoader.zip_to_coco_df_gluon(r[0], unzip_path=unzip_location)
                if not isinstance(df, pd.DataFrame):
                    raise Exception("Data is not a pandas DataFrame")

            elif type == "IM_SEGMENT":
                df = DataLoader.zip_to_segmentation_df_gluon(r[0], unzip_path=unzip_location)
                if not isinstance(df, pd.DataFrame):
                    raise Exception("Data is not a pandas DataFrame")
                
                X, y = DataLoader.zip_to_segmentation_np(r[0], unzip_path=unzip_location)
                if not isinstance(X, np.ndarray) or not isinstance(y, np.ndarray):
                    raise Exception("Data is not a numpy array")
                if len(X.shape) != 4:
                    raise Exception("Data is not a 4D numpy array")
                if not X.shape[0] == y.shape[0]:
                    raise Exception("X and y have different number of samples")
            
            elif type == "IM_CLASS":
                df = DataLoader.zip_to_class_df_gluon(r[0], unzip_path=unzip_location)
                if not isinstance(df, pd.DataFrame):
                    raise Exception("Data is not a pandas DataFrame")
                
                X, y, _ = DataLoader.zip_to_class_np(r[0], unzip_path=unzip_location)
                if not isinstance(X, np.ndarray) or not isinstance(y, np.ndarray):
                    raise Exception("Data is not a numpy array")
                if len(X.shape) != 4:
                    raise Exception("Data is not a 4D numpy array")
                if not X.shape[0] == y.shape[0]:
                    raise Exception("X and y have different number of samples")

    # =================================================================================================================================#
    # ---------------------------------------------------------------------------------------------------------------------------------#
    # =================================================================================================================================#
    @parameterized.expand([
        #("workspace", "dataset", "file_save_location", data_type, problem_type, target),
        ("13b4787c3e454649aa05a4cd680edc37", "986f2e837ca44f3e8c0ee7d2dc0c4287",
          "./data/sedar_raw/zip", "tabular", "classification", "Type"),
        ("13b4787c3e454649aa05a4cd680edc37", "324ea420125d4167a76151b62368c4ad",
          "./data/sedar_raw/zip", "image", "segmentation", None),
        ("13b4787c3e454649aa05a4cd680edc37", "513c6b1ee46b478c8e0925a098d2f387",
          "./data/sedar_raw/zip", "image", "classification", None),
    ])
    def test_sedar_caafe(self, workspace_id, dataset_id, file_save_location, data_type, problem_type, target):
        base_url = "http://192.168.220.107:5000"
        email = "admin"
        password = "admin"

        sedar = SedarAPI(base_url)
        sedar.connection.logger.setLevel("INFO")
        sedar.login(email, password)

        with self.subTest(workspace_id=workspace_id, dataset_id=dataset_id, file_save_location=file_save_location, data_type=data_type, problem_type=problem_type, target=target):
            DataLoader = SedarDataLoader(sedar)
            DataLoader.CAAFEFeatureEngineering(
                workspace_id, 
                dataset_id, 
                file_save_location,
                data_type,
                problem_type,
                target,
                dataset_description='No description',
                iterations=1
            )

# =================================================================================================================================#
if __name__ == '__main__':
    unittest.main()