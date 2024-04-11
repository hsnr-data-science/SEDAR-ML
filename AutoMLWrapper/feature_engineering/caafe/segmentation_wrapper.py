from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from sklearn.model_selection import RepeatedKFold
from sklearn.metrics import accuracy_score, roc_auc_score, jaccard_score
from sklearn.model_selection import train_test_split

from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, concatenate, Conv2DTranspose

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


class CAAFEImageSegmentor(BaseEstimator, ClassifierMixin):
    
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
        
        return self.fit(
            X, y, dataset_description, **kwargs
        )

    def fit(
        self, X, y, dataset_description, disable_caafe=False
    ):
       
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
        self.classes_ = np.unique(y)

        self.base_classifier = self.simple_unet_model(X.shape[1:], len(self.classes_), 'iou')
        self.base_classifier.fit(X, y, epochs=5, verbose=1)

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
        iou , old_iou = [], []

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
            return e, None, None
        from contextlib import contextmanager
        import sys, os
        with open(os.devnull, "w") as devnull:
            old_stdout = sys.stdout
            sys.stdout = devnull
            try:
                result_old = self.evaluate_dataset_segmentation(
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
                result_extended = self.evaluate_dataset_segmentation(
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
        old_iou += [result_old["iou"]]
        iou += [result_extended["iou"]]
        return None, iou, old_iou, 
    


    def get_prompt(
        self,
        ds,
        dataset_info,
        iterative=1,
        data_description_unparsed=None,
         **kwargs
    ):
        how_many = (
            "up to 10 useful transformations. As many transformations as useful for downstream segmentation model, but as few as necessary to reach good performance."
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
            
            This code was written by an expert data scientist working to improve segmentation results. It is a snippet of code that does feature engineering on the dataset based on the available images.
            Number of samples (images) in training dataset "X": {ds[1].shape[0]}. Shape of each image: {ds[1].shape[1:]}.
                
            This code is intended to enhance the dataset by generating additional features or transformations that are beneficial for a downstream segmentation algorithm, such as a U-Net.
            You may consider applying various image transformations, normalizations, and other operations that can extract or highlight important characteristics of the images, or augment the dataset to improve the model's ability to generalize.

            When generating new features or transformations, consider the following:
            - Spatial Consistency: Ensure that transformations maintain spatial relationships, as segmentation requires precise localization of objects or regions.
            - Contextual Information: Generate features or use transformations that preserve or highlight contextual information within the image.
            - Augmentation: Generate additional samples by applying random transformations to the images. This can increase the size of the dataset and improve the model's ability to generalize.
            - Edge Preservation: Consider transformations that preserve or highlight edges, as these are often crucial for accurate segmentation.

            Ensure that all generated features or transformations are based on existing image data and are compatible with the downstream segmentation model. Some points on what this means:
            - Transformations that change the image dimensions must be applied to all images in the dataset. A dataset with images of different sizes cannot be used for training a segmentation model.
            - The datasets variable name is X and shall not be changed. The target variable name is y and shall not be changed. 
            - If additional entities are added to X, i.e., through augmentation, y must be extended accordingly.

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

            The segmentation model will be trained on the dataset enriched with the generated features or transformations and evaluated on a separate holdout set. The primary evaluation metric is the Intersection over Union (IoU), and the objective is to select the code that yields the best performance.

            Note: Limit the use of image processing functions to those available in cv2, PIL, and skimage libraries. You are also allowed to use numpy and sklearn functions. You must import any libraries that you want to use.
            Focus on efficient and effective feature engineering to enhance the dataset meaningfully.

            Code formatting for each transformation (do not forget that the first two lines are comments):
            ```python
            # (name and short description of the transformation)
            # Usefulness: (Description why such a transformation can be useful for segmentation, according to dataset description and image characteristics.)
            (Some code applying the transformation to the image data X)
            ```end

            Each codeblock generates {how_many}.
            Each codeblock ends with ```end and starts with "```python"

            Before writing this Codeblock, consider again: The datasets name MUST be X again after the codeblock.
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
            e, iou, old_iou = self.execute_and_evaluate_code_block(
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
            
            improvement = np.nanmean(iou) - np.nanmean(old_iou)

            add_feature = True
            add_feature_sentence = "The code was executed and changes to ´X´ were kept."
            if improvement  <= 0:
                add_feature = False
                add_feature_sentence = f"The last code changes to ´X´ were discarded. (Improvement: {improvement})"

            display_method(
                "\n"
                + f"*Iteration {i}*\n"
                + f"```python\n{self.format_for_display(code)}\n```\n"
                + f"Performance before adding features IOU {np.nanmean(old_iou):.3f}.\n"
                + f"Performance after adding features IOU {np.nanmean(iou):.3f}.\n"
                + f"{add_feature_sentence}\n"
                + f"\n"
            )

            if len(code) > 10:
                messages += [
                    {"role": "assistant", "content": code},
                    {
                        "role": "user",
                        "content": textwrap.dedent(f"""Performance after adding feature IOU {np.nanmean(iou):.3f}. {add_feature_sentence}
                                    Next codeblock:
                                    """),
                    },
                ]
            if add_feature:
                full_code += code

        return full_code, prompt, messages



    def simple_unet_model(self, input_shape, num_classes, metric = "accuracy"):

        inputs = tf.keras.Input(input_shape)
        if metric == "accuracy":
            pass
        elif metric == "iou":
            if num_classes == 2:
                metric = tf.keras.metrics.BinaryIoU()
            else:
                metric = tf.keras.metrics.MeanIoU(num_classes=num_classes)

        c1 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(inputs)
        p1 = MaxPooling2D((2, 2))(c1)

        c2 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p1)
        p2 = MaxPooling2D((2, 2))(c2)

        c3 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p2)

        u4 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(c3)
        merge4 = concatenate([u4, c2])
        c4 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(merge4)

        u5 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(c4)
        merge5 = concatenate([u5, c1])
        c5 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(merge5)

        if num_classes == 2:
            outputs = Conv2D(1, (1, 1), activation='sigmoid')(c5)
        else:
            outputs = Conv2D(num_classes, (1, 1), activation='softmax')(c5)

        model = tf.keras.Model(inputs=[inputs], outputs=[outputs])
        
        if num_classes == 2:
            model.compile(optimizer='adam', loss='binary_crossentropy', metrics=[metric])
        else:
            model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=[metric])
        

        return model
    
    def evaluate_dataset_segmentation(
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
        model = self.simple_unet_model(X_train.shape[1:], num_classes, metric = "iou")

        model.fit(X_train, y_train, epochs=5, verbose=1, validation_data=(X_valid, y_valid))

        y_pred = model.predict(X_valid)
        if num_classes == 2:
            y_pred = y_pred > 0.5
            iou = jaccard_score(y_valid.flatten(), y_pred.flatten(), average='binary')
        else:
            iou = jaccard_score(y_valid.flatten(), y_pred.flatten(), average='macro')

        method_str = method if type(method) == str else "small unet"
        return {
            "iou": float(iou),
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

    def predict(self, X, num_classes = None):       
        proba = self.predict_proba(X)
        if num_classes == 2 or len(self.classes_) == 2:
            proba = proba > 0.5
        return proba


    def performance_before_run(self, X, y):
        num_classes = len(np.unique(y))
        self.base_classifier = self.simple_unet_model(X.shape[1:], num_classes, 'iou')
        self.base_classifier.fit(X, y, epochs=5, verbose=1)
        
        predictions = self.predict(X, num_classes)
        self.base_classifier = None
        return predictions

    def run_llm_code_numpy(self, code: str, X: np.ndarray, y: np.ndarray) -> np.ndarray:
       
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
