from ..Configuration import Configuration
try:
    import autosklearn
except ImportError as e:
    print(f"WARNING AutoSklearn could not be imported. It might not be available in this environment. Err: \n {e}.")

class AutoSklearnConfig(Configuration):
    __slots__ = []
    #---------------------------------------------------------------------------------------------#
    def __init__(self, config_path: str):
        super().__init__(config_path)

    #---------------------------------------------------------------------------------------------#
    def get_params_constructor_by_key(self, key):
        params =  self.map_user_params('constructor', model_type=key, user_hyperparameters=self.user_hyperparameters)
        """
            memory_limit default 3072 MB produces error.
            n_jobs < 5 produces error.
        """

        if params.get('n_jobs', 0) < 5 and params.get('memory_limit', 0) < 6000:
            params['memory_limit'] = 6000
            params['n_jobs'] = 5
        
        metric = params.get('metric', "")
        if metric == '':
            pass
        if metric == 'accuracy':
            params['metric'] = autosklearn.metrics.accuracy
        elif metric == 'balanced_accuracy':
            params['metric'] = autosklearn.metrics.balanced_accuracy
        elif metric == 'f1':
            params['metric'] = autosklearn.metrics.f1
        elif metric == 'f1_macro':
            params['metric'] = autosklearn.metrics.f1_macro
        elif metric == 'f1_micro':
            params['metric'] = autosklearn.metrics.f1_micro
        elif metric == 'f1_weighted':
            params['metric'] = autosklearn.metrics.f1_weighted
        elif metric == 'roc_auc':
            params['metric'] = autosklearn.metrics.roc_auc
        elif metric == 'precision':
            params['metric'] = autosklearn.metrics.precision
        elif metric == 'precision_macro':
            params['metric'] = autosklearn.metrics.precision_macro
        elif metric == 'precision_weighted':
            params['metric'] = autosklearn.metrics.precision_weighted
        elif metric == 'average_precision':
            params['metric'] = autosklearn.metrics.average_precision
        elif metric == 'recall':
            params['metric'] = autosklearn.metrics.recall
        elif metric == 'recall_macro':
            params['metric'] = autosklearn.metrics.recall_macro
        elif metric == 'recall_weighted':
            params['metric'] = autosklearn.metrics.recall_weighted
        elif metric == 'log_loss':
            params['metric'] = autosklearn.metrics.log_loss
        elif metric == 'r2':
            params['metric'] = autosklearn.metrics.r2
        elif metric == 'mean_squared_error':
            params['metric'] = autosklearn.metrics.mean_squared_error
        elif metric == 'mean_absolute_error':
            params['metric'] = autosklearn.metrics.mean_absolute_error
        elif metric == 'median_absolute_error':
            params['metric'] = autosklearn.metrics.median_absolute_error
        else:
            params['metric'] = autosklearn.metrics.accuracy

        return params
    #---------------------------------------------------------------------------------------------#
    def get_params_fit_by_key(self, key):
        params = self.map_user_params(func_type='fit', model_type=key, user_hyperparameters=self.user_hyperparameters)        

        return params