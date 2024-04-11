from ..Configuration import Configuration

class AutoKerasConfig(Configuration):
    __slots__ = []
    #---------------------------------------------------------------------------------------------#
    def __init__(self, config_path: str):
        super().__init__(config_path)

    #---------------------------------------------------------------------------------------------#
    def get_params_constructor_by_key(self, key):
        params=self.map_user_params(func_type='constructor', model_type=key, user_hyperparameters=self.user_hyperparameters)

        """ Error if objective is not set and accuracy is not in evaluation_metric.
        """
        if type(params.get('objective', None)) == str:
            if params.get('objective', "") == "" and 'accuracy' not in params.get('evaluation_metric', []):
                if type(params['evaluation_metric']) == list:
                    params['evaluation_metric'].append('accuracy')
                else :
                    params['evaluation_metric'] = [params.get('evaluation_metric',0), 'accuracy']

        return params


    #---------------------------------------------------------------------------------------------#
    def get_params_fit_by_key(self, key):
        return self.map_user_params(func_type='fit', model_type=key, user_hyperparameters=self.user_hyperparameters)