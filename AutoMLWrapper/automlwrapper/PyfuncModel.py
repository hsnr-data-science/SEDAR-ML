import mlflow.pyfunc

class PyfuncModel(mlflow.pyfunc.PythonModel):
    def __init__(self, library_name, model_object):
        #self.predictor_class = automl_predictor_class
        #self._artifacts = artifacts
        self._library_name = library_name
        self.model = model_object

    def load_context(self, context):
        #if self._library_name == "autogluon":
            #self.model = self.predictor_class.load(context.artifacts.get("predictor_path"))        
        pass

    def predict(self, context, x):
        return self.model.predict(x)