try:
    from autogluon.tabular import TabularPredictor
    from autogluon.multimodal import MultiModalPredictor
    from autogluon.timeseries import TimeSeriesPredictor
except ImportError as e:
    print(f"WARNING AutoGluon could not be mported. It might not b available in this environment. Err: \n {e}.")


import os
import yaml
import mlflow.pyfunc
from datetime import datetime
import pandas as pd
from PIL import Image
import numpy as np
import json

from ..AutoMLLibrary import AutoMLLibrary
from ..ModelInfo import ModelInfo
from ..PyfuncModel import PyfuncModel
from .AutoGluonConfig import AutoGluonConfig

class AutoGluonWrapper(AutoMLLibrary):
    __slots__ = ['autogluon_problem_type', 'is_hpo']
    #---------------------------------------------------------------------------------------------#
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.autogluon_problem_type = None
        self.is_hpo = False
        self.config = AutoGluonConfig(os.path.join(os.path.dirname(__file__), 'AutoGluonConfig.yaml'))
        if not os.path.exists(os.path.join(os.getcwd(), 'AutoMLOutput')):
            os.makedirs(os.path.join(os.getcwd(), 'AutoMLOutput'))
        self.output_path = os.path.join(os.getcwd(),
                                         f'AutoMLOutput/autogluon{datetime.timestamp(datetime.now())}')
    #---------------------------------------------------------------------------------------------#
    def _map_problem_type(self):

        if self.data_type == 'tabular':
            if self.task_type == 'classification':
                if self.problem_type == 'binary':
                    self.autogluon_problem_type = 'binary'
                elif self.problem_type == 'multiclass':
                    self.autogluon_problem_type = 'multiclass'
                else:
                    raise Exception(f'Unknown problem type {self.problem_type} for {self.task_type}')
            
            elif self.task_type == 'regression':
                if self.problem_type == 'regression':
                    self.autogluon_problem_type = 'regression'
                elif self.problem_type == 'quantile':
                    self.autogluon_problem_type = 'quantile'
                else:
                    raise Exception(f'Unknown problem type {self.problem_type} for {self.task_type}')
        
        elif self.data_type == 'image':
            if self.task_type == 'classification':
                if self.problem_type == 'binary':
                    self.autogluon_problem_type = 'binary'
                elif self.problem_type == 'multiclass':
                    self.autogluon_problem_type = 'multiclass'
                else:
                    raise Exception(f'Unknown problem type {self.problem_type} for {self.task_type}')
            
            elif self.task_type == 'few-shot-classification':
                if self.problem_type == 'binary':
                    self.autogluon_problem_type = 'few_shot_classification'
                elif self.problem_type == 'multiclass':
                    self.autogluon_problem_type = 'few_shot_classification'
                else:
                    raise Exception(f'Unknown problem type {self.problem_type} for {self.task_type}')
            
            elif self.task_type == 'zero-shot-classification':
                if self.problem_type == 'binary':
                    self.autogluon_problem_type = 'zero_shot_image_classification'
                elif self.problem_type == 'multiclass':
                    self.autogluon_problem_type = 'zero_shot_image_classification'
                else:
                    raise Exception(f'Unknown problem type {self.problem_type} for {self.task_type}')
            
            elif self.task_type == 'regression':
                if self.problem_type == 'regression':
                    self.autogluon_problem_type = 'regression'
                else:
                    raise Exception(f'Unknown problem type {self.problem_type} for {self.task_type}')
            
            elif self.task_type == 'object-detection':
                if self.problem_type == 'object-detection':
                    self.autogluon_problem_type = 'object_detection'
                else:
                    raise Exception(f'Unknown problem type {self.problem_type} for {self.task_type}')

            elif self.task_type == 'zero-shot-object-detection':
                if self.problem_type == 'object-detection': 
                    self.autogluon_problem_type = 'open_vocabulary_object_detection'
                else:
                    raise Exception(f'Unknown problem type {self.problem_type} for {self.task_type}')
            
            elif self.task_type == 'similarity':
                if self.problem_type == 'similarity':
                    self.autogluon_problem_type = 'image_similarity'
                else:
                    raise Exception(f'Unknown problem type {self.problem_type} for {self.task_type}')
            
            elif self.task_type == 'segmentation':
                self.autogluon_problem_type = 'semantic_segmentation'
            

        elif self.data_type == 'text':
            if self.task_type == 'classification':
                if self.problem_type == 'binary':
                    self.autogluon_problem_type = 'binary'
                elif self.problem_type == 'multiclass':
                    self.autogluon_problem_type = 'multiclass'
                elif self.problem_type == 'few-shot':
                    self.autogluon_problem_type = 'few_shot_classification'
                else:
                    raise Exception(f'Unknown problem type {self.problem_type} for {self.task_type}')
            
            elif self.task_type == 'similarity':
                if self.problem_type == 'similarity':
                    self.autogluon_problem_type = 'text_similarity'
                else:
                    raise Exception(f'Unknown problem type {self.problem_type} for {self.task_type}')
        
        elif self.data_type == 'timeseries':
            if self.task_type == 'forecast':
                if self.problem_type == 'forecast':
                    pass
        
        else:
            raise Exception(f'Unknown data type {self.data_type}')
    
     #---------------------------------------------------------------------------------------------#
    def data_preprocessing(self, data, target_column = 'label'):
        
        if self.custom_data_preprocessing_func is not None:
            data = self.custom_data_preprocessing_func(data)
            return data
        return data
    
    #---------------------------------------------------------------------------------------------#
    def _train_model(self, data, val_data,  target_column, user_hyperparameters: dict = {}):
        
        if type(data) not in [pd.DataFrame]:
            raise ValueError(f'data must be of type pandas DataFrame, but got {type(data)}')  

        self.is_hpo = True if 'hpo' in user_hyperparameters.get('preset', '') else False      
        
        self.config.map_hyperparameters(user_hyperparameters)
        self._map_problem_type()
        
        data = self.data_preprocessing(data, target_column)
        val_data = self.data_preprocessing(val_data, target_column)
            
        if self.autogluon_problem_type in ['semantic_segmentation']:
            sample_data = data.iloc[0][target_column]
        else:
            sample_data = data
                
        if self.data_type == 'image' or self.data_type == 'text':
            self.model = MultiModalPredictor(
                sample_data_path=sample_data,
                label=target_column,
                path=self.output_path,
                problem_type=self.autogluon_problem_type,
                **(self.config.get_params_constructor_by_key('MultiModalPredictor') or {})
            )
            self.log_model_type = self.config._get_mlflow_details('MultiModalPredictor').get('__log_model_type', {})

        elif self.data_type == 'tabular':
            self.model = TabularPredictor(
                label=target_column,
                path=self.output_path,
                problem_type=self.autogluon_problem_type,
                **(self.config.get_params_constructor_by_key('TabularPredictor') or {})
            )
            self.log_model_type = self.config._get_mlflow_details('TabularPredictor').get('__log_model_type', {})

        elif self.data_type == 'timeseries':
            self.model = TimeSeriesPredictor(
                target=target_column,
                path=self.output_path,
                **(self.config.get_params_constructor_by_key('TimeSeriesPredictor') or {})
            )
            self.log_model_type = self.config._get_mlflow_details('TimeSeriesPredictor').get('__log_model_type', {})
        

        if self.autogluon_problem_type in ['open_vocabulary_object_detection', 'zero_shot_image_classification']:
            self.eval_output = self._handle_zero_shot(data, target_column)
            return self.eval_output
            
        
        self.fit_output = self.model.fit(
            train_data=data,
            tuning_data=val_data,
            **(self.config.get_params_fit_by_key('TimeSeriesPredictor' if self.data_type == 'timeseries' 
                                          else 'TabularPredictor' if self.data_type == 'tabular' 
                                          else 'MultiModalPredictor') or {})
        )

    #---------------------------------------------------------------------------------------------#
    def _evaluate_model(self, test_data, target_column, **kwargs):
        if self.autogluon_problem_type in ['open_vocabulary_object_detection', 'zero_shot_image_classification']:
            print('Zero-shot models will not be evaluated. The predictions from the training data have been returned.')
            return self.eval_output
        
        self.eval_output = self.model.evaluate(
            data = self.data_preprocessing(test_data, target_column),    
            **kwargs
        )

    #---------------------------------------------------------------------------------------------#
    def _create_model_info(self, n_models: int = 1):

        model_infos = []

        #if self.is_hpo:
        #    trial_scores = self._get_hpo_trial_scores()
            

        for i in range(n_models):
            if self.data_type in ['tabular', 'timeseries']:
                model_info = self._get_info_from_fit_summary(i)
            elif self.data_type in ['image', 'text']:
                #if self.is_hpo:
                #    model_info = self._get_info_from_hpo(i, trial_scores)
                    
                #else:
                    model_info = self._get_info_from_config_yaml()
                    
                    if n_models > 1:
                        print(f'Only one model is available. The model info for this model will be returned.')
                        model_infos.append(model_info)
                        break
            
            model_infos.append(model_info)
        
        return model_infos

    #---------------------------------------------------------------------------------------------#
    def  _get_info_from_fit_summary(self, n_th_model):

        dict_summary = self.model.fit_summary(0)
        df_leaderboard = self.model.leaderboard(silent=True)
        try:
            model_name = df_leaderboard.loc[n_th_model, 'model']
        except KeyError:
            raise ValueError(f'No more than {n_th_model-1} models found. Please try at max {len(df_leaderboard)} models.')
                    
        mi = ModelInfo(
            model_name = dict_summary['model_types'].get(model_name, None),
            model_library = 'autogluon',
            model_object = PyfuncModel('autogluon', self.model),
            model_path = self.output_path + 'models/',
            model_params_dict = dict_summary['model_hyperparams'].get(model_name, {}),
            model_metrics_dict = {
                'val_score': df_leaderboard.loc[df_leaderboard['model'] == model_name, 'score_val'].values[0],
                **({self.eval_output} if self.eval_output is not None else {})
            },
            model_tags_dict = {
                'data_type': self.data_type,
                'task_type': self.task_type,
                'problem_type': self.problem_type,
            },
        )

        return mi
        
    #---------------------------------------------------------------------------------------------#
    def _get_info_from_config_yaml(self, path = None):
        if path is None:
            path = self.output_path

        from_config_yaml = False
        from_params_json = False
        from_hparams_yaml = False

        try:
            with open(path + '/config.yaml') as file:
                out_config = yaml.load(file, Loader=yaml.FullLoader)
                from_config_yaml = True
        except FileNotFoundError:
            try:
                with open(path + '/params.json') as file:
                    out_config = json.load(file)
                    from_params_json = True
            except FileNotFoundError:
                out_config = {}
        
        try:
            with open(path + '/hparams.yaml') as file:
                hparams = yaml.load(file, Loader=yaml.FullLoader)
                from_hparams_yaml = True
        except FileNotFoundError:
            hparams = {}

        if from_config_yaml:
            try:
                models = out_config['model']['names']
                models[0]
            except Exception:
                models = ['neural net ' + self.data_type + ' ' + self.task_type]

            model_config = out_config['model'].get(models[0], {})
            data_config = out_config.get('data', {})
            opt_config = out_config.get('optimization', {})
            env_config = out_config.get('env', {})

            model_name = models[0] + ':' + model_config.get('checkpoint_name', '')
            train_transforms = model_config.get('train_transforms', [])
            val_transforms = model_config.get('val_transforms', [])

            hparams = {**hparams, **({'max_epochs': opt_config['max_epochs']} if 'max_epochs' in opt_config else {})}
            hparams = {**hparams, **({'patience': opt_config['patience']} if 'patience' in opt_config else {})}
            hparams = {**hparams, **({'batch_size': env_config['batch_size']} if 'batch_size' in opt_config else {})}
            hparams = {**hparams, **({'train_transforms': train_transforms} if train_transforms else {})}
            hparams = {**hparams, **({'val_transforms': val_transforms} if val_transforms else {})}
            
            if from_hparams_yaml:
                files = [ path + '/config.yaml', path + '/hparams.yaml']
            else:
                files = [ path + '/config.yaml',]

        elif from_params_json:
            models = out_config['root']['model.names']
            model_name = models[0] + ':' + out_config['root']['model.' + models[0] + '.checkpoint_name']

            hparams = {**hparams, **({'max_epochs': out_config['optimization.max_epochs']} if 'optimization.max_epochs' in out_config else {})}
            hparams = {**hparams, **({'optim_type': out_config['optimization.optim_type']} if 'optimization.optim_type' in out_config else {})}
            hparams = {**hparams, **({'batch_size': out_config['env.batch_size']} if 'env.batch_size' in out_config else {})}
            hparams = {**hparams, **({'train_transforms': out_config['optimization.learning_rate']} if 'optimization.learning_rate' in out_config else {})}
            
            if from_hparams_yaml:
                files = [ path + '/config.yaml', path + '/hparams.yaml']
            else:
                files = [ path + '/config.yaml',]
        
        metrics = self.model.fit_summary(0)
        if self.eval_output is not None:
            metrics = {**metrics, **self.eval_output}

        mi = ModelInfo(
            model_name = model_name,
            model_library = 'autogluon',
            model_object = PyfuncModel('autogluon', self.model),
            model_path = self.output_path + 'models/',
            model_params_dict = {k: v for k, v in hparams.items() if v is not None},
            model_metrics_dict = metrics,
            model_files_as_path_list = files,
            model_tags_dict = {
                'data_type': self.data_type,
                'task_type': self.task_type,
                'problem_type': self.problem_type,
            }
        )

        return mi 
    
    #---------------------------------------------------------------------------------------------#
    def _get_hpo_trial_scores(self):
        trial_scores = {}
        folders = [f for f in os.listdir(self.output_path) if os.path.isdir(os.path.join(self.output_path, f))]
        
        for folder in folders:
            with open(self.output_path + f'/{folder}/progress.csv') as file:
                progress = pd.read_csv(file)

        
            val_score_col = [col for col in progress.columns if col.startswith('val_')][0]      
            series = progress.loc[progress[val_score_col].nlargest(1).index[-1]]

            trial_id = series['trial_id']
            trial_scores[trial_id] = series[val_score_col]
            
        # i.e. score is loss, lower is better        
        reverse = False if list(trial_scores.values())[0] < 0 else True
        trial_scores = {k: v for k, v in sorted(trial_scores.items(), key=lambda item: item[1], reverse=reverse)}

    #---------------------------------------------------------------------------------------------#
    def _get_info_from_hpo(self, n_th_model, trial_scores):
        
        trial_id = list(trial_scores.keys())[n_th_model]

        path = self.output_path + f'/{trial_id}/'

        model_info = self._get_info_from_config_yaml(path)

        return model_info
        
        
    #---------------------------------------------------------------------------------------------#
    def _handle_zero_shot(self, data, target_column):
        print('Zero-shot models will not be trained. The Model will now make predictions on the data and return the results.')
        
        targets = self.config.get_params_predict_by_key('TimeSeriesPredictor' if self.problem_type == 'timeseries' 
                                          else 'TabularPredictor' if self.problem_type == 'tabular' 
                                          else 'MultiModalPredictor')['candidate_data'] 
        
        targets = self._convert_to_zero_shot_prompt(targets, 
                                                    classification=True if self.autogluon_problem_type in ['zero_shot_image_classification'] else False)  

        label_col = 'prompt' if self.autogluon_problem_type == 'zero_shot_image_classification' else 'text'       
           
        if target_column in data.columns:
            data = data.drop(columns=[target_column])
        else:
            pass      

        data[label_col] = [targets] * len(data)

        if self.autogluon_problem_type == 'zero_shot_image_classification':
            ev = self.model.predict(
                data['image'].to_list(),
                {"text": targets},
                as_pandas=True
                )
            
        elif self.autogluon_problem_type == 'open_vocabulary_object_detection':
            ev = self.model.predict(data, as_pandas=False)
        
        return ev

    #---------------------------------------------------------------------------------------------#
    def _convert_to_zero_shot_prompt(self, targets, classification=False):
        
        # return a list of targets
        if classification:
            if isinstance(targets, list):
                return targets 
            elif isinstance(targets, str):
                if ',' in targets:
                    return targets.split(',')
                elif '.' in targets:
                    return targets.split('.')
                else:
                    return [targets]
        
        #return a period-separated string
        else:
            if isinstance(targets, list):
                return ['. '.join(targets)]
            elif isinstance(targets, str):
                if ',' in targets:
                    return [targets.replace(',', '. ')]
                else:
                    return [targets]