from typing import Any, Optional, Dict

from .AutoGluon.AutoGluonWrapper import AutoGluonWrapper
from .AutoKeras.AutoKerasWrapper import AutoKerasWrapper
from .AutoSklearn.AutoSklearnWrapper import AutoSklearnWrapper
from .ModelInfo import ModelInfo
from .MLflowHandler import MLflowHandler

class AutoMLWrapper:
    __slots__ = ['__library', '__library_name', '__is_initialized', '__out_path', '__extra_allowed_hyperparameters']
    __properties__ = ['_library', '_library_name', '_is_initialized', '_out_path', '_extra_allowed_hyperparameters']
    #---------------------------------------------------------------------------------------------#
    def __init__(self, library_name: str) -> None:
        self.__library = None
        self.__library_name = library_name
        self.__is_initialized = False
        self.__out_path = None
        self.__extra_allowed_hyperparameters = None
        
    #---------------------------------------------------------------------------------------------#
    def SetOutputDirectory(self, outputDirectory: str) -> None:
        """Set the output directory for the AutoML process.
        """
        self.__out_path = outputDirectory

    #---------------------------------------------------------------------------------------------#
    def SetCustomDataPreprocessing(self, custom_preprocessing_func):
        """
        Set a custom data preprocessing function that will be applied before model training.
        """
        self.__library._set_custom_data_preprocessing(custom_preprocessing_func)

    #---------------------------------------------------------------------------------------------#
    def AllowExtraHyperparameters(self, extra_allowed_hyperparameters: dict):
        """
        Set extra hyperparameters for each fuunction-type (fit, predict, ...) as a dict list.
        """
        # return if not dict of lists
        if not isinstance(extra_allowed_hyperparameters, dict):
            print("extra_allowed_hyperparameters must be a dict of lists")
            return
        if not all([isinstance(v, list) for v in extra_allowed_hyperparameters.values()]):
            print("extra_allowed_hyperparameters must be a dict of lists")
            return

        self.__extra_allowed_hyperparameters = extra_allowed_hyperparameters

    #---------------------------------------------------------------------------------------------#
    @property
    def _library(self) -> Any:
        return self.__library
    
    @_library.setter
    def _library(self, _) -> None:
        if self.__library_name == "autosklearn":
            self.__library = AutoSklearnWrapper()

        elif self.__library_name == "autokeras":
            self.__library = AutoKerasWrapper()

        elif self.__library_name == "autogluon":
            self.__library = AutoGluonWrapper()

        else:
            raise ValueError("Invalid library name")
    
    #---------------------------------------------------------------------------------------------#
    @property
    def _library_name(self) -> str:
        return self.__library_name
    
    @_library_name.setter
    def _library_name(self, library_name: str) -> None:
        raise NotImplementedError("This property is read-only")
    
    #---------------------------------------------------------------------------------------------#
    @property
    def _is_initialized(self) -> bool:
        return self.__is_initialized

    @_is_initialized.setter
    def _is_initialized(self, is_initialized: bool) -> None:
        raise NotImplementedError("This property is read-only")

    #---------------------------------------------------------------------------------------------#
    @property
    def _out_path(self) -> str:
        return self.__out_path
    
    @_out_path.setter
    def _out_path(self, out_path: str) -> None:
        ### add validation ??? ###
        self.__out_path = out_path
    
    #---------------------------------------------------------------------------------------------#
    @property
    def _extra_allowed_hyperparameters(self) -> dict:
        return self.__extra_allowed_hyperparameters
    
    @_extra_allowed_hyperparameters.setter
    def _extra_allowed_hyperparameters(self, extra_allowed_hyperparameters: dict) -> None:
        raise NotImplementedError("This property is read-only")

    #---------------------------------------------------------------------------------------------#
    def Initialize(self, data_sample: Any, target_column: str, task_type: Optional[str] = None, data_type: Optional[str] = None, problem_type: Optional[str] = None) -> None:
        """Initialize the AutoML process. Use for determining the task type, data type, and problem type beforehand.
        """

        if self.__library == None:
            self._library = self.__library_name 

        if task_type is not None:
            self.__library._set_task_type(task_type)
        else:
            if data_sample is not None:
                self.__library._infer_task_type(data_sample, target_column)
        
        if data_type is not None:
            self.__library._set_data_type(data_type)
        else:
            if data_sample is not None:
                self.__library._infer_data_type(data_sample, target_column)

        if problem_type is not None:
            self.__library._set_problem_type(problem_type)
        else:
            if data_sample is not None:
                self.__library._infer_problem_type(data_sample, target_column)

        if self.__out_path is not None:
            self.__library._set_out_path(self.__out_path)

        if data_sample is not None:
             self.__is_initialized = True
        else:
            self.__is_initialized = False

        
        if self._extra_allowed_hyperparameters is not None:
            self.__library._set_extra_allowed_hyperparameters(self._extra_allowed_hyperparameters)

    #---------------------------------------------------------------------------------------------#
    def Train(self, train_data: Any, validation_data: Any = None, target_column: str = 'label', task_type: Optional[str] = None, data_type: Optional[str] = None, 
              problem_type: Optional[str] = None, hyperparameters: Optional[Dict[str, Any]] = {}) -> None:
        """ Invoke the underlying AutoML library to train a model.
        """

        if not self.__is_initialized:
            self.Initialize(train_data, target_column, task_type, data_type, problem_type)

        return self.__library._train_model(train_data, validation_data, target_column, hyperparameters)

    #---------------------------------------------------------------------------------------------#
    def Evaluate(self, test_data: Any,  target_column: str, **kwargs) -> float:
        if self.__library is None:
            raise ValueError("You must call 'train' before 'evaluate'")
        
        self.__library._evaluate_model(test_data,  target_column, **kwargs)
        return self.__library.eval_output

    # #---------------------------------------------------------------------------------------------#
    # def Output(self, nBestModels: int = 1, outputForMLFlow : bool = True):
    #     if self.__library is None:
    #         raise ValueError("You must call 'train' before 'output'")

    #     if outputForMLFlow:
    #         return self.__library._create_model_info(nBestModels)
    #     else:
    #         pass
    
    #---------------------------------------------------------------------------------------------#
    def MlflowUploadBest(self, user_tags: dict):
        best_model_info = self.__library._create_model_info(1)[0]  

        if not best_model_info:
            return

        mlflow_handler = MLflowHandler(best_model_info, user_tags)
        return mlflow_handler.log_to_mlflow()    

    #---------------------------------------------------------------------------------------------#
    def MlflowUploadTopN(self, n: int, user_tags: dict):
        top_n_models_info = self.__library._create_model_info(n)

        run_ids = []
        for i, model_info in enumerate(top_n_models_info):
   
            mlflow_handler = MLflowHandler(model_info, user_tags)
            run_ids.append(mlflow_handler.log_to_mlflow())

        return run_ids 

