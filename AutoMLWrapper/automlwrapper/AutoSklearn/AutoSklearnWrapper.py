try:
    from autosklearn.classification import AutoSklearnClassifier
    from autosklearn.regression import AutoSklearnRegressor
except ImportError as e:
    print(f"WARNING AutoSklearn could not be mported. It might not b available in this environment. Err: \n {e}.")

import os
from datetime import datetime
import pandas as pd

from ..AutoMLLibrary import AutoMLLibrary
from ..ModelInfo import ModelInfo
from ..PyfuncModel import PyfuncModel
from .AutoSklearnConfig import AutoSklearnConfig

class AutoSklearnWrapper(AutoMLLibrary):
    __slots__ = []
    #---------------------------------------------------------------------------------------------#
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.config = AutoSklearnConfig(os.path.join(os.path.dirname(__file__), 'AutoSklearnConfig.yaml'))
        
        
        if not os.path.exists(os.path.join(os.getcwd(), 'AutoMLOutput')):
            os.makedirs(os.path.join(os.getcwd(), 'AutoMLOutput'))

        self.output_path = os.path.join(os.getcwd(),
                                         f'AutoMLOutput/autosklearn{datetime.timestamp(datetime.now())}')

    #---------------------------------------------------------------------------------------------#
    def _data_preprocessing(self, data, val_data, target_column):
        
        if self.custom_data_preprocessing_func is not None:
            data = self.custom_data_preprocessing_func(data)
            return data
        
        if val_data is not None:
            X_train, y_train = self.separate(data, target_column, type='pandas')
            X_val, y_val = self.separate(val_data, target_column, type='pandas')
            return {'X': X_train, 'y': y_train, 'X_val': X_val, 'y_val': y_val}
        
        X_train, y_train, X_test, y_test = self.split_and_separate(data, target_column, ratio=0.2, type='pandas')
        return {'X': X_train, 'y': y_train, 'X_test': X_test, 'y_test': y_test}
    
    #---------------------------------------------------------------------------------------------#
    def _train_model(self, data, val_data, target_column : str, user_hyperparameters : dict = {}):
        
        if type(data) not in [pd.DataFrame]:
            raise ValueError(f'data must be of type pandas DataFrame, but got {type(data)}')
        
        self.config.map_hyperparameters(user_hyperparameters)

        if self.task_type == 'classification':
            self.model = AutoSklearnClassifier(
                tmp_folder=self.output_path,
                delete_tmp_folder_after_terminate=False,
                **(self.config.get_params_constructor_by_key('AutoSklearnClassifier') or {})
            )
            self.log_model_type = self.config._get_mlflow_details('AutoSklearnClassifier').get('__log_model_type', {})

        elif self.task_type == 'regression':
            self.model = AutoSklearnRegressor(
                tmp_folder=self.output_path,
                delete_tmp_folder_after_terminate=False,
                **(self.config.get_params_constructor_by_key('AutoSklearnRegressor') or {})
            )
            self.log_model_type = self.config._get_mlflow_details('AutoSklearnRegressor').get('__log_model_type', {})

        self.fit_output = self.model.fit(
            **(self._data_preprocessing(data, val_data, target_column)),
            **(self.config.get_params_fit_by_key('AutoSklearnClassifier' if self._set_task_type == 'classification' 
                                          else 'AutoSklearnRegressor' if self._set_task_type == 'regression'
                                          else None) or {})
            )
    
    #---------------------------------------------------------------------------------------------#
    def _evaluate_model(self, test_data, target_column, **kwargs):
        if type(test_data) not in [pd.DataFrame]:
            raise ValueError(f'data must be of type pandas DataFrame, but got {type(test_data)}')


        X_test, y_test = self._separate_x_y_df(test_data, target_column=target_column)

        if kwargs.get('predict', False) == True:
            self.eval_output = self.model.predict(X_test)
        else:
        
            self.eval_output = self.model.score(
                X_test, y_test
            )
            return self.eval_output
    

    #---------------------------------------------------------------------------------------------#
    def _create_model_info(self, n: int = 1):

        best_models_info = []
        df_leaderboard = self.fit_output.leaderboard(detailed=True)
        df_leaderboard.insert(0, 'model_id', df_leaderboard.index)
        df_leaderboard.reset_index(drop=True, inplace=True)       

        dict_model_info = self.fit_output.show_models()

        for i in range(n):
            try:
                model_id = df_leaderboard.loc[i, 'model_id']
            except KeyError:
                raise ValueError(f'No {n} models found. Please try at max {len(df_leaderboard)} models.')
                
            model_info = ModelInfo(
                **(self._get_info_by_id(model_id, df_leaderboard, dict_model_info))
            )
            
            best_models_info.append(model_info)

        return best_models_info

    #---------------------------------------------------------------------------------------------#
    def _get_info_by_id(self, model_id, leaderboard, model_info):
        model_info_by_id = model_info[model_id]
        model_data_preprocessor_name = leaderboard.loc[leaderboard['model_id'] == model_id, 'data_preprocessors'].values[0]
        model_feature_preprocessor_name = leaderboard.loc[leaderboard['model_id'] == model_id, 'feature_preprocessors'].values[0]

        _tmp_model_key = 'classifier' if self.task_type == 'classification' else 'regressor' if self.task_type == 'regression' else ValueError('No task type set.')

        hparams = model_info_by_id['sklearn_' + _tmp_model_key].get_params()
        model_info_args = {
            'model_name': model_info_by_id['sklearn_' + _tmp_model_key].__class__.__name__,
            'model_library': 'autosklearn',
            'model_object': PyfuncModel('autosklearn', self.model),
            'model_path': None,
            'model_params_dict': {k: v for k, v in hparams.items() if v is not None},
            'model_metrics_dict': {
                'model_loss': model_info_by_id['cost'],
            },
            'model_tags_dict':{
                'model_data_preprocessor_name': model_data_preprocessor_name,
                'model_feature_preprocessor_name': model_feature_preprocessor_name,
            },
        }

        return model_info_args