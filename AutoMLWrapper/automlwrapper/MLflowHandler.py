import mlflow
from typing import Dict, Any
from .ModelInfo import ModelInfo

class MLflowHandler:
    #---------------------------------------------------------------------------------------------#
    def __init__(self, model_info: ModelInfo, user_tags: Dict[str, Any]):
        self.model_info = model_info
        self.user_tags = user_tags

    #---------------------------------------------------------------------------------------------#
    def log_to_mlflow(self):

        run_ids = []
        with mlflow.start_run() as run:
            for key, value in self.user_tags.items():
                mlflow.set_tag(key, str(value))

            model_info_dict = self.model_info.to_dict()

            for key, value in model_info_dict['tags'].items():
                mlflow.set_tag(key, str(value))
            for key, value in model_info_dict['params'].items():
                mlflow.log_param(key, value)
            for key, value in model_info_dict['metrics'].items():
                mlflow.log_metric(key, value)
            for key, value in model_info_dict['images'].items():
                mlflow.log_image(value, key)
            for path in model_info_dict['files']:
                mlflow.log_artifact(path)

            mlflow.pyfunc.log_model("model", python_model=self.model_info.model_object)

            run_ids.append(run.info.run_id)
        
        return run_ids