
SEDAR_SCHEMA_USTRUCTURED = "UNSTRUCTURED"
SEDAR_SCHEMA_STRUCTURED = "STRUCTURED"


# task type
OBJECT_DETECTION = "object-detection"
REGRESSION = "regression"
CLASSIFICATION = "classification"
SEGMENTATION = "segmentation"
ZERO_OBJECT_DETECTION = "zero-shot-object-detection"
SIMILARITY = "similarity"
FEW_SHOT_CLASSIFICATION = "few-shot-classification"
ZERO_SHOT_CLASSIFICATION = "zero-shot-classification"
FORECAST = "forecast"

# problem type
BINARY = "binary"
MULTICLASS = "multiclass"
OBJECT_DETECTION_PROBLEM = "object-detection"
SIMILARITY_PROBLEM = "similarity"
FORECAST_PROBLEM = "forecast"
MULTILABEL = "multilabel"
MULTIOUTPUT = "multioutput"
REGRESSION_PROBLEM = "regression"
QUANTILE = "quantile"

# data types
TABULAR = "tabular"
TIME_SERIES = "time_series"
IMAGE = "image"
TEXT = "text"


def isObjectDetection(type: str):
    return type.lower() in [OBJECT_DETECTION, 'obj', 'object-detection', 'object_detection', 'object detection']

def isTaskObjectDetection(task: str):
    return task.lower() in [OBJECT_DETECTION, 'obj', 'object-detection', 'object_detection', 'object detection']

def isTaskSegmentation(task: str):
    return task.lower() in [SEGMENTATION, 'seg', 'image-segmentation', 'image_segmentation','image segmentation']

def isTaskClassification(task: str):
    return task.lower() in [CLASSIFICATION, 'cls', 'image-classification', 'image_classification', 'image classification']

def isTaskRegression(task: str):
    return task.lower() in [REGRESSION, 'reg']

def isTaskSimilarity(task: str):
    return task.lower() in [SIMILARITY, 'sim']

def isTaskForecast(task: str):
    return task.lower() in [FORECAST, 'fc']

def isTaskZeroShotClassification(task: str):
    return task.lower() in [ZERO_SHOT_CLASSIFICATION, 'zsc', 'zero_shot_classification', 'zero shot classification']

def isTaskFewShotClassification(task: str):
    return task.lower() in [FEW_SHOT_CLASSIFICATION, 'fsc', 'few_shot_classification', 'few shot classification']

def isTaskZeroObjectDetection(task: str):
    return task.lower() in [ZERO_OBJECT_DETECTION, 'zsod', 'zero_shot_object_detection', 'zero shot object detection']

def isProblemBinary(problem: str):
    return problem.lower() in [BINARY, 'bin', 'binary']

def isProblemMulticlass(problem: str):
    return problem.lower() in [MULTICLASS, 'multi', 'multiclass']

def isProblemObjectDetection(problem: str):
    return problem.lower() in [OBJECT_DETECTION_PROBLEM, 'obj', 'object-detection', 'object_detection', 'object detection']

def isProblemSimilarity(problem: str):
    return problem.lower() in [SIMILARITY_PROBLEM, 'sim', 'similarity']

def isProblemForecast(problem: str):        
    return problem.lower() in [FORECAST_PROBLEM, 'fc', 'forecast']

def isProblemMultilabel(problem: str):
    return problem.lower() in [MULTILABEL, 'ml', 'multi-label', 'multi_label', 'multi label']

def isProblemMultioutput(problem: str):
    return problem.lower() in [MULTIOUTPUT, 'mo', 'multi-output', 'multi_output', 'multi output']

def isProblemRegression(problem: str):
    return problem.lower() in [REGRESSION_PROBLEM, 'reg', 'regression']

def isProblemQuantile(problem: str):
    return problem.lower() in [QUANTILE, 'quant', 'quantile']

def isDataTypeTabular(data_type: str):
    return data_type.lower() in [TABULAR, 'tab', 'tabular']

def isDataTypeTimeSeries(data_type: str):
    return data_type.lower() in [TIME_SERIES, 'ts', 'time-series', 'time_series', 'time series']

def isDataTypeImage(data_type: str):
    return data_type.lower() in [IMAGE, 'img', 'image']

def isDataTypeText(data_type: str):
    return data_type.lower() in [TEXT, 'txt', 'text']


def isImgFile(path: str):
    return path.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif'))


