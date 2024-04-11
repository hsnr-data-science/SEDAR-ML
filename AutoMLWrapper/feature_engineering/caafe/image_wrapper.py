from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense


from .preprocessing import (
    make_datasets_numeric,
    split_target_column,
    make_dataset_numeric,
)
from .caafe_evaluate import (
    evaluate_dataset,
)

from .data import get_X_y
from .caafe import generate_features
from .metrics import auc_metric, accuracy_metric
import tensorflow as tf
import copy
import pandas as pd
import numpy as np
from typing import Optional
import pandas as pd
import textwrap
import openai
import ast


class CAAFEImageClassifier(BaseEstimator, ClassifierMixin):
    """
    A classifier that uses the CAAFE algorithm to generate features and a base classifier to make predictions.

    Parameters:
    base_classifier (object, optional): The base classifier to use. If None, a default TabPFNClassifier will be used. Defaults to None.
    optimization_metric (str, optional): The metric to optimize during feature generation. Can be 'accuracy' or 'auc'. Defaults to 'accuracy'.
    iterations (int, optional): The number of iterations to run the CAAFE algorithm. Defaults to 10.
    llm_model (str, optional): The LLM model to use for generating features. Defaults to 'gpt-3.5-turbo'.
    n_splits (int, optional): The number of cross-validation splits to use during feature generation. Defaults to 10.
    n_repeats (int, optional): The number of times to repeat the cross-validation during feature generation. Defaults to 2.
    """
    def __init__(
        self,
        base_classifier: Optional[object] = None,
        optimization_metric: str = "accuracy",
        iterations: int = 10,
        llm_model: str = "gpt-3.5-turbo",
        n_splits: int = 10,
        n_repeats: int = 2,
    ) -> None:
        self.base_classifier = base_classifier
        self.llm_model = llm_model
        self.iterations = iterations
        self.optimization_metric = optimization_metric
        self.n_splits = n_splits
        self.n_repeats = n_repeats
        self.code = ""

    def fit_images(self, X, y, dataset_description, **kwargs):
        """
        Fit the classifier to a pandas DataFrame.

        Parameters:
        df (pandas.DataFrame): The DataFrame to fit the classifier to.
        dataset_description (str): A description of the dataset.
        target_column_name (str): The name of the target column in the DataFrame.
        **kwargs: Additional keyword arguments to pass to the base classifier's fit method.
        """
        # feature_columns = list(df.drop(columns=[target_column_name]).columns)

        # X, y = (
        #     df.drop(columns=[target_column_name]).values,
        #     df[target_column_name].values,
        # )

        return self.fit(
            X, y, dataset_description, **kwargs
        )

    def fit(
        self, X, y, dataset_description, disable_caafe=False
    ):
        """
        Fit the model to the training data.

        Parameters:
        -----------
        X : np.ndarray
            The training data features.
        y : np.ndarray
            The training data target values.
        dataset_description : str
            A description of the dataset.
        feature_names : List[str]
            The names of the features in the dataset.
        target_name : str
            The name of the target variable in the dataset.
        disable_caafe : bool, optional
            Whether to disable the CAAFE algorithm, by default False.
        """
        self.dataset_description = dataset_description
        self.X_ = X
        self.y_ = y
        ds = [
            "dataset",
            X,
            y,
            [],
            {},
            dataset_description,
        ]

        if disable_caafe:
            self.code = ""
        else:
            self.code, prompt, messages = self.generate_features_image(
                ds,
                model=self.llm_model,
                iterative=self.iterations,
                metric_used=auc_metric,
                iterative_method=self.base_classifier,
                display_method="markdown",
                n_splits=self.n_splits,
                n_repeats=self.n_repeats,
            )

        X, y = self.run_llm_code_numpy(
            self.code,
            X,
            y
        )

        # Store the classes seen during fit
        self.classes_ = unique_labels(y)

        self.base_classifier = self.make_model_conv2d(X.shape[1:], len(self.classes_))
        self.base_classifier.fit(X, to_categorical(y, len(self.classes_)), epochs=5, verbose=1)

        return self
    

    def format_for_display(self, code):
        code = code.replace("```python", "").replace("```", "").replace("<end>", "")
        return code

    def generate_code(self, messages, model):
        if model == "skip":
            return ""

        completion = openai.ChatCompletion.create(
            model=model,
            messages=messages,
            stop=["```end"],
            temperature=0.5,
            max_tokens=500,
        )
        code = completion["choices"][0]["message"]["content"]
        code = code.replace("```python", "").replace("```", "").replace("<end>", "")
        return code
    
    def execute_and_evaluate_code_block(
            self,
            code_history,
            code_new,
            n_splits,
            n_repeats,
            ds,
            iterative_method,
            metric_used,
            display_method
        ):
        old_accs, old_rocs, accs, rocs = [], [], [], []

        X=ds[1]
        y=ds[2]
        indices = np.arange(len(X))
        train_indices, valid_indices = train_test_split(indices, test_size=0.2, random_state=None)
        
        X_train, X_valid = X[train_indices], X[valid_indices]
        y_train, y_valid = y[train_indices], y[valid_indices]
                    
        X_train_extended = copy.deepcopy(X_train)
        X_valid_extended = copy.deepcopy(X_valid)
        y_train_extended = copy.deepcopy(y_train)
        y_valid_extended = copy.deepcopy(y_valid)


        try:
            X_train, y_train = self.run_llm_code_numpy(
                code_history,
                X_train,
                y_train
            )
            X_valid, y_valid = self.run_llm_code_numpy(
                code_history,
                X_valid,
                y_valid
            )
            X_train_extended, y_train_extended = self.run_llm_code_numpy(
                code_history + "\n" + code_new,
                X_train_extended,
                y_train_extended
            )
            X_valid_extended, y_valid_extended = self.run_llm_code_numpy(
                code_history + "\n" + code_new,
                X_valid_extended,
                y_valid_extended
            )
        except Exception as e:
            display_method(f"Error in code execution. {type(e)} {e}")
            display_method(f"```python\n{self.format_for_display(code_new)}\n```\n")
            return e, None, None, None, None
        from contextlib import contextmanager
        import sys, os
        with open(os.devnull, "w") as devnull:
            old_stdout = sys.stdout
            sys.stdout = devnull
            try:
                result_old = self.evaluate_dataset_image(
                    X_train=X_train,
                    X_valid=X_valid,
                    y_train=y_train,
                    y_valid=y_valid,
                    prompt_id="XX",
                    name=ds[0],
                    method=iterative_method,
                    metric_used=metric_used,
                    seed=0,
                )
                result_extended = self.evaluate_dataset_image(
                    X_train=X_train_extended,
                    X_valid=X_valid_extended,
                    y_train=y_train_extended,
                    y_valid=y_valid_extended,
                    prompt_id="XX",
                    name=ds[0],
                    method=iterative_method,
                    metric_used=metric_used,
                    seed=0,
                )
            finally:
                sys.stdout = old_stdout
        old_accs += [result_old["roc"]]
        old_rocs += [result_old["acc"]]
        accs += [result_extended["roc"]]
        rocs += [result_extended["acc"]]
        
        return  None, rocs, accs, old_rocs, old_accs
    


    def get_prompt(
            self,
            ds,
            dataset_info,
            iterative=1,
            data_description_unparsed=None,
             **kwargs
        ):
        how_many = (
            "up to 10 useful transformations. As many transformations as useful for downstream classifier, but as few as necessary to reach good performance."
            if iterative == 1
            else "exactly one useful transformation"
        )

        return textwrap.dedent(f"""
            The image data is loaded and in memory. Each image has the following characteristics:
            
            {dataset_info['pixel_stats']}
            {dataset_info['class_distribution_desc']}
            {dataset_info['class_imbalance_ratio']}
            {dataset_info['sample_targets']}
            
            Description of the dataset: "{data_description_unparsed}"
            
            This code was written by an expert data scientist working to improve predictions. It is a snippet of code that does feature engineering on the dataset based on the available images.
            Number of samples (images) in training dataset "X": {ds[1].shape[0]}. Shape of each image: {ds[1].shape[1:]}.
                
            This code is intended to enhance the dataset by generating additional features that are beneficial for a downstream classification algorithm, such as a Convolutional Neural Network (CNN).
            You may consider applying various image transformations, normalizations, and other operations that can extract or highlight important characteristics of the images.

            When generating new features, consider the following:
            - Transformations: Apply geometric and color space transformations to reveal different perspectives of the image data. Utilize functions from cv2, PIL, or skimage for operations like resizing, rotating, color space conversion, and filtering.
            - Normalization: Standardize the pixel values to ensure that the input data has a consistent scale. This can improve the convergence of the classifier during training.
            - Feature Extraction: Extract meaningful attributes from the images, such as edges, textures, or patterns. Techniques like edge detection, texture analysis, or histogram analysis can be useful.
            - Aggregation: Combine different features or statistics to create composite features that might capture complex patterns or relationships in the data.
            - Augmentation: Generate additional samples by applying random transformations to the images. This can increase the size of the dataset and improve the generalization of the classifier.

            Ensure that all generated features are based on existing image data and are compatible with the downstream classifier. Some points on what this means:
            - transformations that change the image dimensions must be applied to all images in the dataset. A dataset with images of different sizes cannot be used for training a CNN.
            - the datasets variable name is X and shall not be changed. The target variable name is y and shall not be changed. 
            - if additional entities are added to X, i. e. through augmentation, y must be extended accordingly.

            It is paramount that the transformed dataset is named X again! Here is an example of the naming convention:
            --- begin example ---
            # either concatenate the new features to the original dataset
            # only if the dimensions of the images are not changed! For resized images, do not do this!
            X = np.concatenate((X, *transformed array*))
            y = np.concatenate((y, y))   # dont forget to extend y accordingly!

            # or create a new dataset, what you think is best
            X = *transformed array*
            # notice that in both cases the dataset is named X again!
            --- end example ---

            The classifier will be trained on the dataset enriched with the generated features and evaluated on a separate holdout set. The primary evaluation metric is accuracy, and the objective is to select the code that yields the best performance.

            Note: Limit the use of image processing functions to those available in cv2, PIL, and skimage libraries. You are also allowed to use numpy and sklearn functions. You must import any libraries that you want to use.
            Focus on efficient and effective feature engineering to enhance the dataset meaningfully.

            Code formatting for each transformation (do not forget that the first two lines are comments):
            ```python
            # (name and short description of the transformation)
            # Usefulness: (Description why such a transformation can be usefull for classification, according to dataset description and image characteristics.)
            (Some code applying the transformation to the image data X)
            ```end

            Each codeblock generates {how_many}.
            Each codeblock ends with ```end and starts with "```python"

            Before writing this Codeblock, cosnider again: The datasets name MUST be X again after the codeblock.
            Codeblock:
            """)

    def build_prompt_from_numpy(self, ds, iterative=1):
        dataset_description = ds[-1]
        X, y = ds[1], ds[2]
    
        image_shape = X.shape[1:] 
        pixel_stats = f"Image dimensions (HxWxC): {image_shape}\n"
        pixel_stats += f"Pixel value range: {X.min()} to {X.max()}\n"
        pixel_stats += f"Mean pixel value: {X.mean():.2f}, Stddev: {X.std():.2f}\n"

        unique, counts = np.unique(y, return_counts=True)
        class_distribution = dict(zip(unique, counts))
        class_distribution_desc = f"Class distribution: {class_distribution}\n"
        class_imbalance_ratio = f"Class imbalance ratio: {max(counts) / min(counts):.2f}\n"
        sample_targets = f"Sample targets: {y[:10]}\n"

        dataset_info = {
            "pixel_stats": pixel_stats,
            "class_distribution_desc": class_distribution_desc,
            "class_imbalance_ratio": class_imbalance_ratio,
            "sample_targets": sample_targets,
        }
    

        prompt = self.get_prompt(
            ds,
            dataset_info,
            data_description_unparsed=dataset_description,
            iterative=iterative,
        )

        return prompt


    def generate_features_image(
        self,
        ds,
        model="gpt-3.5-turbo",
        just_print_prompt=False,
        iterative=1,
        metric_used=None,
        iterative_method="logistic",
        display_method="markdown",
        n_splits=10,
        n_repeats=2,
        ):

        if display_method == "markdown":
            from IPython.display import display, Markdown

            display_method = lambda x: display(Markdown(x))
        else:
            display_method = print

        assert (
            iterative == 1 or metric_used is not None
        ), "metric_used must be set if iterative"

        prompt = self.build_prompt_from_numpy(ds,  iterative=iterative)

        if just_print_prompt:
            code, prompt = None, prompt
            return code, prompt, None

        messages = [
            {
                "role": "system",
                "content": "You are an expert datascientist assistant solving Kaggle problems. You answer only by generating code. Answer as concisely as possible.",
            },
            {
                "role": "user",
                "content": prompt,
            },
        ]
        display_method(f"*Dataset description:*\n {ds[-1]}")

        n_iter = iterative
        full_code = ""

        i = 0
        while i < n_iter:
            try:
                code = self.generate_code(messages, model)
            except Exception as e:
                display_method("Error in LLM API." + str(e))
                i = i + 1
                continue
            i = i + 1
            e, rocs, accs, old_rocs, old_accs = self.execute_and_evaluate_code_block(
                full_code,
                code,
                n_splits,
                n_repeats,
                ds,
                iterative_method,
                metric_used,
                display_method
            )
            if e is not None:
                messages += [
                    {"role": "assistant", "content": code},
                    {
                        "role": "user",
                        "content": f"""Code execution failed with error: {type(e)} {e}.\n Code: ```python{code}```\n Generate next feature (fixing error?):
                                    ```python
                                    """,
                    },
                ]
                continue

            improvement_roc = np.nanmean(rocs) - np.nanmean(old_rocs)
            improvement_acc = np.nanmean(accs) - np.nanmean(old_accs)

            add_feature = True
            add_feature_sentence = "The code was executed and changes to the data were kept."
            if improvement_roc + improvement_acc <= 0:
                add_feature = False
                add_feature_sentence = f"The last code changes to the data were discarded. (Improvement: {improvement_roc + improvement_acc})"

            display_method(
                "\n"
                + f"*Iteration {i}*\n"
                + f"```python\n{self.format_for_display(code)}\n```\n"
                + f"Performance before adding features ROC {np.nanmean(old_rocs):.3f}, ACC {np.nanmean(old_accs):.3f}.\n"
                + f"Performance after adding features ROC {np.nanmean(rocs):.3f}, ACC {np.nanmean(accs):.3f}.\n"
                + f"Improvement ROC {improvement_roc:.3f}, ACC {improvement_acc:.3f}.\n"
                + f"{add_feature_sentence}\n"
                + f"\n"
            )

            if len(code) > 10:
                messages += [
                    {"role": "assistant", "content": code},
                    {
                        "role": "user",
                        "content": textwrap.dedent(f"""Performance after adding feature ROC {np.nanmean(rocs):.3f}, ACC {np.nanmean(accs):.3f}. {add_feature_sentence}
                                    Next codeblock:
                                    """),
                    },
                ]
            if add_feature:
                full_code += code

        return full_code, prompt, messages



    def make_model_conv2d(self, input_shape, num_classes):
        model = Sequential([
            Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape),
            MaxPooling2D(pool_size=(2, 2)),
            Conv2D(64, kernel_size=(3, 3), activation='relu'),
            MaxPooling2D(pool_size=(2, 2)),
            Flatten(),
            Dense(128, activation='relu'),
            Dense(num_classes, activation='softmax')
        ])
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

        return model
    
    def evaluate_dataset_image(
        self,
        X_train: np.ndarray,
        X_valid: np.ndarray,
        y_train: np.ndarray,
        y_valid: np.ndarray,
        prompt_id,
        name,
        method,
        metric_used,
        max_time=300,
        seed=0,
    ):
        np.random.seed(seed)
        tf.random.set_seed(seed)

        num_classes = len(np.unique(y_train))
        y_train_cat = to_categorical(y_train, num_classes)
        y_valid_cat = to_categorical(y_valid, num_classes)

        model = self.make_model_conv2d(X_train.shape[1:], num_classes)

        model.fit(X_train, y_train_cat, epochs=5, verbose=1, validation_data=(X_valid, y_valid_cat))

        _, accuracy = model.evaluate(X_valid, y_valid_cat, verbose=0)
        y_pred = model.predict(X_valid)
        y_pred_label = np.argmax(y_pred, axis=1)

        acc = accuracy_score(y_valid, y_pred_label)
        roc = roc_auc_score(y_valid_cat, y_pred, multi_class='ovo') if num_classes > 2 else roc_auc_score(y_valid, y_pred[:, 1])

        method_str = method if type(method) == str else "cnn"
        return {
            "acc": float(acc),
            "roc": float(roc),
            "prompt": prompt_id,
            "seed": seed,
            "name": name,
            "size": len(X_train),
            "method": method_str,
            "max_time": max_time,
            "feats": X_train.shape[-1],
        }


    def apply_code(self, X, y):
        X, y = self.run_llm_code_numpy(
            self.code,
            X,
            y
        )
        return X, y
    
    def predict_preprocess(self, X):
        """
        Helper function for preprocessing the image data before making predictions.

        Parameters:
        X (numpy.ndarray): The image data to make predictions on.

        Returns:
        numpy.ndarray: The preprocessed image data.
        """

        X, _ = self.run_llm_code_numpy(
            self.code,
            X,
            np.zeros((X.shape[0], 1))
        )
        return X


    def predict_proba(self, X):
        X_preprocessed = self.predict_preprocess(X)
        proba = self.base_classifier.predict(X_preprocessed)
        return proba

    def predict(self, X):       
        proba = self.predict_proba(X)
        predictions = np.argmax(proba, axis=1)
        return predictions


    def performance_before_run(self, X, y):
        num_classes = len(np.unique(y))
        self.base_classifier = self.make_model_conv2d(X.shape[1:], num_classes)
        self.base_classifier.fit(X, to_categorical(y, num_classes), epochs=5, verbose=1)
        
        predictions = self.predict(X)
        self.base_classifier = None
        return predictions

    def run_llm_code_numpy(self, code: str, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Executes the given code on the given image data (numpy arrays) and returns the resulting data.

        Parameters:
        code (str): The code to execute.
        X (numpy.ndarray): The image data to execute the code on.
        y (numpy.ndarray): The target data.

        Returns:
        numpy.ndarray: The resulting data after executing the code.
        """

        try:
            loc = {}
            X = copy.deepcopy(X)
            y = copy.deepcopy(y)
            
            access_scope = {"X": X, "y": y, 'np': np, 'pd': pd}
            parsed = ast.parse(textwrap.dedent(code))
            self.check_ast(parsed)
            #exec(compile(parsed, filename="<ast>", mode="exec"), access_scope, loc)
            exec(code, access_scope, access_scope)
            X = copy.deepcopy(access_scope["X"])
            y = copy.deepcopy(access_scope["y"])

        except Exception as e:
            print("Code could not be executed", e)
            raise e

        return X, y
    

    def check_ast(self, node: ast.AST) -> None:
        """
        Checks if the given AST node is allowed.

        Parameters:
        node (ast.AST): The AST node to check.

        Raises:
        ValueError: If the AST node is not allowed.
        """
        allowed_nodes = {
            ast.Module,
            ast.Expr,
            ast.Load,
            ast.BinOp,
            ast.UnaryOp,
            ast.Add,
            ast.Sub,
            ast.Mult,
            ast.Div,
            ast.FloorDiv,
            ast.Mod,
            ast.Pow,
            ast.USub,
            ast.UAdd,
            ast.Num,
            ast.Str,
            ast.Bytes,
            ast.List,
            ast.Tuple,
            ast.Dict,
            ast.Name,
            ast.Call,
            ast.Attribute,
            ast.keyword,
            ast.Subscript,
            ast.Index,
            ast.Slice,
            ast.ExtSlice,
            ast.Assign,
            ast.AugAssign,
            ast.NameConstant,
            ast.Compare,
            ast.Eq,
            ast.NotEq,
            ast.Lt,
            ast.LtE,
            ast.Gt,
            ast.GtE,
            ast.Is,
            ast.IsNot,
            ast.In,
            ast.NotIn,
            ast.And,
            ast.Or,
            ast.BitOr,
            ast.BitAnd,
            ast.BitXor,
            ast.Invert,
            ast.Not,
            ast.Constant,
            ast.Store,
            ast.If,
            ast.IfExp,
            # These nodes represent loop structures. If you allow arbitrary loops, a user could potentially create an infinite loop that consumes system resources and slows down or crashes your system.
            ast.For,
            ast.While,
            ast.Break,
            ast.Continue,
            ast.Pass,
            ast.Assert,
            ast.Return,
            ast.FunctionDef,
            ast.ListComp,
            ast.SetComp,
            ast.DictComp,
            ast.GeneratorExp,
            ast.Await,
            # These nodes represent the yield keyword, which is used in generator functions. If you allow arbitrary generator functions, a user might be able to create a generator that produces an infinite sequence, potentially consuming system resources and slowing down or crashing your system.
            ast.Yield,
            ast.YieldFrom,
            ast.Lambda,
            ast.BoolOp,
            ast.FormattedValue,
            ast.JoinedStr,
            ast.Set,
            ast.Ellipsis,
            ast.expr,
            ast.stmt,
            ast.expr_context,
            ast.boolop,
            ast.operator,
            ast.unaryop,
            ast.cmpop,
            ast.comprehension,
            ast.arguments,
            ast.arg,
            ast.Import,
            ast.ImportFrom,
            ast.alias,
        }

        allowed_packages = {"numpy", "pandas", "sklearn", "cv2", "PIL", "skimage"}

        allowed_funcs = {
            "sum": sum,
            "min": min,
            "max": max,
            "abs": abs,
            "round": round,
            "len": len,
            "str": str,
            "int": int,
            "float": float,
            "bool": bool,
            "list": list,
            "dict": dict,
            "set": set,
            "tuple": tuple,
            "enumerate": enumerate,
            "zip": zip,
            "range": range,
            "sorted": sorted,
            "reversed": reversed,
            # Add other functions you want to allow here.
        }

        allowed_attrs = {
            # NP
            "array",
            "arange",
            "values",
            "linspace",
            "column_stack",
            # PD
            "mean",
            "sum",
            "contains",
            "where",
            "min",
            "max",
            "median",
            "std",
            "sqrt",
            "pow",
            "iloc",
            "cut",
            "qcut",
            "inf",
            "nan",
            "isna",
            "map",
            "reshape",
            "shape",
            "split",
            "var",
            "codes",
            "abs",
            "cumsum",
            "cumprod",
            "cummax",
            "cummin",
            "diff",
            "repeat",
            "index",
            "log",
            "log10",
            "log1p",
            "slice",
            "exp",
            "expm1",
            "pow",
            "pct_change",
            "corr",
            "cov",
            "round",
            "clip",
            "dot",
            "transpose",
            "T",
            "astype",
            "copy",
            "drop",
            "dropna",
            "fillna",
            "replace",
            "merge",
            "append",
            "join",
            "groupby",
            "resample",
            "rolling",
            "expanding",
            "ewm",
            "agg",
            "aggregate",
            "filter",
            "transform",
            "apply",
            "pivot",
            "melt",
            "sort_values",
            "sort_index",
            "reset_index",
            "set_index",
            "reindex",
            "shift",
            "extract",
            "rename",
            "tail",
            "head",
            "describe",
            "count",
            "value_counts",
            "unique",
            "nunique",
            "idxmin",
            "idxmax",
            "isin",
            "between",
            "duplicated",
            "rank",
            "to_numpy",
            "to_dict",
            "to_list",
            "to_frame",
            "squeeze",
            "add",
            "sub",
            "mul",
            "div",
            "mod",
            "columns",
            "loc",
            "lt",
            "le",
            "eq",
            "ne",
            "ge",
            "gt",
            "all",
            "any",
            "clip",
            "conj",
            "conjugate",
            "round",
            "trace",
            "cumprod",
            "cumsum",
            "prod",
            "dot",
            "flatten",
            "ravel",
            "T",
            "transpose",
            "swapaxes",
            "clip",
            "item",
            "tolist",
            "argmax",
            "argmin",
            "argsort",
            "max",
            "mean",
            "min",
            "nonzero",
            "ptp",
            "sort",
            "std",
            "var",
            "str",
            "dt",
            "cat",
            "sparse",
            "plot"
            # cv2
            "imread",
            "resize", 
            "cvtColor", 
            "GaussianBlur", 
            "Canny",
            "findContours",
            "drawContours", 
            "threshold",
            #pil
            "Image.Image.resize",
            "Image.Image.crop",
            "Image.Image.rotate",
            "Image.Image.filter",
            "ImageOps.grayscale",
            "ImageOps.equalize", 
            #skimage
            "skimage.transform.resize",
            "skimage.transform.rotate",
            "skimage.filters.gaussian",
            "skimage.feature.canny",
            "skimage.exposure.equalize_hist",
            "skimage.measure.label",
            "skimage.color.rgb2gray",
            #sklearn
            "sklearn.preprocessing.StandardScaler",
        }

        return None
    
        if type(node) not in allowed_nodes:
            raise ValueError(f"Disallowed code: {ast.unparse(node)} is {type(node)}")

        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id not in allowed_funcs:
                raise ValueError(f"Disallowed function: {node.func.id}")

        if isinstance(node, ast.Attribute) and node.attr not in allowed_attrs:
            raise ValueError(f"Disallowed attribute: {node.attr}")

        if isinstance(node, ast.Import) or isinstance(node, ast.ImportFrom):
            for alias in node.names:
                if alias.name not in allowed_packages:
                    raise ValueError(f"Disallowed package import: {alias.name}")

        for child in ast.iter_child_nodes(node):
            self.check_ast(child)
