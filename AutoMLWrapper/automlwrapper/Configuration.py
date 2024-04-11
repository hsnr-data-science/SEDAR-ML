import yaml

class Configuration:
    __slots__ = ['config', 'user_hyperparameters', 'extra_allowed_hyperparameters', 'used_params']
    #---------------------------------------------------------------------------------------------#
    def __init__(self, config_file_path):
        self.config = self._load_config(config_file_path)
        self.user_hyperparameters = {}
        self.extra_allowed_hyperparameters = {}
        self.used_params = set()

    #---------------------------------------------------------------------------------------------#
    def _load_config(self, config_file_path):
        with open(config_file_path, 'r') as config_file:
            return yaml.safe_load(config_file)
    
    #---------------------------------------------------------------------------------------------#
    def map_hyperparameters(self, user_hyperparameters: dict):
        self.user_hyperparameters = user_hyperparameters
    
    #---------------------------------------------------------------------------------------------#
    def set_extra_allowed_hyperparameters(self, extra_allowed_hyperparameters: dict):
        if not isinstance(extra_allowed_hyperparameters, dict):
            print("extra_allowed_hyperparameters must be a dict")
            return
        self.extra_allowed_hyperparameters = extra_allowed_hyperparameters

    #---------------------------------------------------------------------------------------------#
    def _get_hyperparameter_details(self, func_type: str = None, model_type: str = None):
        """
        Fetches hyperparameters based on the func_type (like 'constructor' or 'fit') 
        and model_type (like 'TabularPredictor').
        If model_type is provided, it fetches hyperparameters specific to that model type.
        """
        return self.config.get(model_type, {}).get('__hyperparameter_details', {}).get(func_type, {})
    
    #---------------------------------------------------------------------------------------------#
    def _get_mlflow_details(self, model_type: str = None):
        return self.config.get(model_type, {}).get('__mlflow', {})
    
    #---------------------------------------------------------------------------------------------#
    def map_user_params(self, func_type: str = None, model_type: str = None, user_hyperparameters: dict = {}):
        """
        Maps user-defined hyperparameters to the hyperparameters defined in the config file.
        returns: dictionary of hyperparameters for application to library functions.
        """

        hyperparameters = {}

        # Retrieve hyperparameter details from configuration
        hp_details = self._get_hyperparameter_details(func_type, model_type)

        for hp_key, hp_value in hp_details.items():
            if hp_key == '__extra_args':
                for extra_hp_name, extra_hp_params in hp_value.items():
                    extra_args = self._map_extra_args(func_type, extra_hp_params, user_hyperparameters)
                    if extra_args:
                        hyperparameters[extra_hp_name] = extra_args
            else:
                user_key = hp_value.get('__user_mapping')
                if user_key in user_hyperparameters:
                    hyperparameters[hp_key] = user_hyperparameters[user_key]
                    self.used_params.add(user_key)
        
        for user_hp_key, user_hp_value in user_hyperparameters.items():
            if self._matches_extra_allowed(user_hp_key, func_type):
                hyperparameters[user_hp_key] = user_hp_value
                self.used_params.add(user_hp_key)

        missing_hyperparameters = self.user_hyperparameters.keys() - self.used_params 
        if len(missing_hyperparameters) > 0 and func_type == 'fit':
            print(f""" ================================================= \n AutoMLWrapper Warning:
            \n The following hyperparameters were not found in the configuration files and were thus ignored: 
            \n {missing_hyperparameters} 
            \n If you want to use these hyperparameters, set their names explicitly with wrapper.AllowExtraHyperparameters().
            \n =================================================""")
        
        return hyperparameters

    #---------------------------------------------------------------------------------------------#
    def _map_extra_args(self, func_type: str, extra_hp_params: dict, user_hyperparameters: dict):
        """
        Helper function to map extra hyperparameters.
        """
        mapped_params = {}
        for param_name, param_details in extra_hp_params.items():
            if param_name.startswith('__'):
                continue
            user_key = param_details.get('__user_mapping')
            if user_key in user_hyperparameters:
                mapped_params[param_name] = user_hyperparameters[user_key]
                self.used_params.add(user_key)
        return mapped_params


    #---------------------------------------------------------------------------------------------#
    def _matches_extra_allowed(self, user_key: str, func_type: str):
        """
        Helper function to check if a user parameter is an extra hyperparameter.
        """
        try:
            if user_key in self.extra_allowed_hyperparameters[func_type]:
                return True
            else:
                return False
        except KeyError as e:
            return False