try:
    import autokeras as ak
    import keras
except ImportError as e:
    print(f"WARNING AutoKeras could not be imported. It might not be available in this environment. Err: \n {e}.")

import pandas as pd
from datetime import datetime
import os
from PIL import Image
import numpy as np
from ..AutoMLLibrary import AutoMLLibrary
from ..ModelInfo import ModelInfo
from ..PyfuncModel import PyfuncModel
from .AutoKerasConfig import AutoKerasConfig
from ..SedarDataLoader import SedarDataLoader


class AutoKerasWrapper(AutoMLLibrary):
    __slots__ = []
    #---------------------------------------------------------------------------------------------#
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.config = AutoKerasConfig(os.path.join(os.path.dirname(__file__), 'AutoKerasConfig.yaml'))
        if not os.path.exists(os.path.join(os.getcwd(), 'AutoMLOutput')):
            os.makedirs(os.path.join(os.getcwd(), 'AutoMLOutput'))
        self.output_path = os.path.join(os.getcwd(),
                                         f'AutoMLOutput/autokeras{datetime.timestamp(datetime.now())}')
        
    #---------------------------------------------------------------------------------------------#
    def _data_preprocessing(self, data, target, as_tuple = False):

        if self.custom_data_preprocessing_func is not None:
            data = self.custom_data_preprocessing_func(data)
            return data
        
        if type(data) in [pd.DataFrame]:
            if self.data_type == 'image' and self.task_type == 'classification':
                x, y = SedarDataLoader.imageDataFrameToNumpyXy(data, target=target)
            else:
                x, y = self.separate(data, target, type='pandas') 
                
        elif type(data) in [tuple]:
            x, y = self.separate(data, target, type='tuple')
        
        if data is None:
            return None
        if as_tuple:
            return (x, y)
        return {'x': x, 'y': y}
    
    #---------------------------------------------------------------------------------------------#
    def _train_model(self, data, val_data, target_column, user_hyperparameters: dict = {}):

        if type(data) not in [pd.DataFrame, tuple]:
            raise ValueError(f'data must be of type pandas DataFrame or 2-tuple of numpy arrays, but got {type(data)}')
        
        self.config.map_hyperparameters(user_hyperparameters)

        if self.data_type == 'text':
            self._train_text(data, val_data, target_column, user_hyperparameters)
        elif self.data_type == 'tabular':
            self._train_structured(data, val_data, target_column, user_hyperparameters)
        elif self.data_type == 'image':
            self._train_image(data, val_data, target_column, user_hyperparameters)


    def _get_params(self, func_type: str, model_type: str):
        return self.config.get_params(func_type, model_type)
    
    #---------------------------------------------------------------------------------------------#
    def _train_structured(self, data, val_data, target_column, user_hyperparameters: dict = {}):
        if self.task_type == "classification":
            self.model = ak.StructuredDataClassifier(
                directory=self.output_path,
                **(self.config.get_params_constructor_by_key('StructuredDataClassifier') or {})
            )
            self.log_model_type = self.config._get_mlflow_details('StructuredDataClassifier').get('__log_model_type', {})

        elif self.task_type == "regression":
            self.model = ak.StructuredDataRegressor(
                directory=self.output_path,
                **(self.config.get_params_constructor_by_key('StructuredDataRegressor') or {})
            )
            self.log_model_type = self.config._get_mlflow_details('StructuredDataRegressor').get('__log_model_type', {})

        self.fit_output = self.model.fit(
            **(self._data_preprocessing(data, target_column)),
            validation_data=self._data_preprocessing(val_data, target_column, as_tuple=True),
            **(self.config.get_params_fit_by_key('StructuredDataClassifier' if self.task_type == 'classification' 
                                          else 'StructuredDataRegressor' if self.task_type == 'regression'
                                          else None) or {})
            )


    #---------------------------------------------------------------------------------------------#
    def _train_image(self, data, val_data, target_column, user_hyperparameters: dict = {}):
        if self.task_type == "classification":
            self.model = ak.ImageClassifier(
                directory=self.output_path,
                **(self.config.get_params_constructor_by_key('ImageClassifier') or {})
            )
            self.log_model_type = self.config._get_mlflow_details('ImageClassifier').get('__log_model_type', {})

        elif self.task_type == "regression":
            self.model = ak.ImageRegressor(
                directory=self.output_path,
                **(self.config.get_params_constructor_by_key('ImageRegressor') or {})
            )
            self.log_model_type = self.config._get_mlflow_details('ImageRegressor').get('__log_model_type', {})

        self.fit_output = self.model.fit(
            **(self._data_preprocessing(data, target_column)),
            validation_data=self._data_preprocessing(val_data, target_column, as_tuple=True),
            **(self.config.get_params_fit_by_key('ImageClassifier' if self.task_type == 'classification' 
                                          else 'ImageRegressor' if self.task_type == 'regression'
                                          else None) or {})
            )
                    
    #---------------------------------------------------------------------------------------------#
    def _train_text(self, data, val_data, target_column, user_hyperparameters: dict = {}):
        if self.task_type == "classification":
            self.model = ak.TextClassifier(
                directory=self.output_path,
                **(self.config.get_params_constructor_by_key('TextClassifier') or {})
            )
            self.log_model_type = self.config._get_mlflow_details('TextClassifier').get('__log_model_type', {})

        elif self.task_type == "regression":
            self.model = ak.TextRegressor(
                **(self.config.get_params_constructor_by_key('TextRegressor') or {})
            )
            self.log_model_type = self.config._get_mlflow_details('TextRegressor').get('__log_model_type', {})

        self.fit_output = self.model.fit(
            **(self._data_preprocessing(data, target_column)),
            validation_data=self._data_preprocessing(val_data, target_column, as_tuple=True),
            **(self.config.get_params_fit_by_key('TextClassifier' if self.task_type == 'classification' 
                                          else 'TextRegressor' if self.task_type == 'regression'
                                          else None) or {})
            )

    #---------------------------------------------------------------------------------------------#
    def _train_timeseries(self, data, val_data, target_column, user_hyperparameters: dict = {}):
        self.model = ak.TimeseriesForecaster(
            directory=self.output_path,
            **(self.config.get_params_constructor_by_key('TimeseriesForecaster') or {})
        )
        self.log_model_type = self.config._get_mlflow_details('TimeseriesForecaster').get('__log_model_type', {})

        self.fit_output = self.model.fit(
            **(self._data_preprocessing(data, target_column)),
            validation_data=self._data_preprocessing(val_data, target_column, as_tuple=True),
            **(self.config.get_params_fit_by_key('TimeseriesForecaster') or {})
            )

    #---------------------------------------------------------------------------------------------#
    def _evaluate_model(self, test_data, target_column, **kwargs):
        if type(test_data) not in [pd.DataFrame, tuple]:
            raise ValueError(f'data must be of type pandas DataFrame or 2-tuple of numpy arrays, but got {type(test_data)}')
      

        self.eval_output = self.model.evaluate(
            **(self._data_preprocessing(test_data, target_column)),
            **kwargs
        )
        return self.eval_output


    #---------------------------------------------------------------------------------------------#
    def _create_model_info(self, n: int = 1):
        best_models_info = []

        if n > 1:
            raise ValueError(f'Library {self.__class__.__name__} does not support more than one model. Please set n=1.')
        
        model = self.model.export_model()
        model_info = ModelInfo(
            **(self._get_info_from_keras_model(model))
        )

        best_models_info.append(model_info)
        
        return best_models_info

    
    #---------------------------------------------------------------------------------------------#
    def _get_info_from_keras_model(self, keras_model):

        model_optimizer = keras_model.optimizer.get_config()
        optimizer_info = {f"optimizer_{key}": value for key, value in model_optimizer.items()}
        metrics_info = {key : value[-1] for key, value in self.fit_output.history.items()}

        keras.utils.plot_model(keras_model, to_file=self.output_path + '/model_img.png',
                                show_shapes=True, show_layer_names=True)
        
        model_img = Image.open(self.output_path + '/model_img.png')

        optimizer_info['epochs'] = len(self.fit_output.epoch)
        
        model_info_args = {
            'model_name': 'neural net ' + self.data_type + ' ' + self.task_type,
            'model_library': 'autokeras',
            'model_object': PyfuncModel('autokeras', keras_model),
            'model_path': None,
            'model_params_dict': {k: v for k, v in optimizer_info.items() if v is not None},
            'model_metrics_dict': metrics_info,
            'model_tags_dict':{},
            'model_imgs_as_pil_dict': {'network_structure.png': model_img},
        }

        return model_info_args

