import sys
import os
import shutil
import zipfile
import logging
from typing import Union, List, Optional, Tuple
import pandas as pd
import numpy as np

try:
    from pyspark.sql.session import SparkSession
except ImportError:
    print("WARNING: PySpark not installed. Some functionalities might not be available.")
    SparkSession = None

try: 
    from pywebhdfs.webhdfs import PyWebHdfsClient
except ImportError:
    print("WARNING: pywebhdfs not installed. Some functionalities might not be available.")
    PyWebHdfsClient = None

from PIL import Image
from sedarapi import SedarAPI
from .constants import *
from sklearn.model_selection import train_test_split


logger = logging.getLogger('automlwrapper')

class SedarDataLoader:
    __slots__ = ['spark_session', 'sedarapi', 'hdfs']

    # --------------------------------------------------------------------------------------------#
    def __init__(self, sedar : SedarAPI):
        self.spark_session = None
        self.sedarapi = sedar

        self.hdfs = PyWebHdfsClient(host='192.168.220.107', port='9870')

    # --------------------------------------------------------------------------------------------#
    def create_spark_session(self):
        jars = ["io.delta:delta-core_2.12:1.0.0"]
 
        configs = {
            #"spark.master": "spark://192.168.220.107:7077",
            "spark.master": "local",
            "spark.app.name": "7777778-c47a-48d0-a3e4-9c91cdd92c3f1",
           # "spark.jars": tiledb_jar,
            "spark.submit.deployMode": "client",
            "spark.jars.packages": ",".join(jars),
            "spark.driver.bindAddress": "0.0.0.0",
            #######################
            "spark.cores.max": "2",
            "spark.executor.memory": "16G",
            "spark.driver.memory": "16G",
            "spark.driver.maxResultSize": "0",
            "spark.pyspark.driver.python": sys.executable,
            #########################
            "spark.executor.heartbeatInterval": "60s",
            "spark.network.timeoutInterval": "300s",
            "spark.network.timeout": "300s",
            "spark.sql.broadcastTimeout": "10000",
            #########################
            "fs.defaultFS": "hdfs://192.168.220.107:9000",
            "dfs.client.use.datanode.hostname": "false",
            "dfs.datanode.use.datanode.hostname": "false",
        }
        builder = SparkSession.builder
        for k, v in configs.items():
            builder = builder.config(k, v)
        spark_session = builder.getOrCreate()

        return spark_session

    # --------------------------------------------------------------------------------------------#
    def spark_df_from_json(self, json, is_local_path=False):
        if not self.spark_session:
            self.spark_session = self.create_spark_session()


        if is_local_path:
            df = self.spark_session.read.json(json)
        else:
            df = self.spark_session.createDataFrame(json)

        return df

    # --------------------------------------------------------------------------------------------#
    def query_data(self, sedar_workspace_id: str, sedar_dataset_id: str,
                   query: Optional[str] = None, file_save_location: Optional[str] = None) -> Union[pd.DataFrame, List[str]]:
        """
        Queries data from a specified SEDAR workspace and dataset. It can return a pandas DataFrame for structured data
        or save unstructured data to a specified location and return the paths.

        Parameters:
        - sedar_workspace_id (str): The ID of the SEDAR workspace.
        - sedar_dataset_id (str): The ID of the SEDAR dataset.
        - query (Optional[str]): An SQL query to run against the dataset if it is structured. Defaults to None.
        - file_save_location (Optional[str]): The directory path to save files if the dataset is unstructured. Defaults to None.

        Returns:
        Union[pd.DataFrame, List[str]]: A pandas DataFrame for structured data or a list of file paths for unstructured data.
        """
        logger.info("Querying data from SEDAR.")

        ws = self.sedarapi.get_workspace(sedar_workspace_id)
        ds = ws.get_dataset(sedar_dataset_id)

        try:
            schema_type = ds.content['schema']['type'].upper()
            query_spark = schema_type in ['STRUCTURED', 'SEMISTRUCTURED']
            query_raw = schema_type in ['UNSTRUCTURED', 'IMAGE']
            
            if not query_spark and not query_raw:
                logger.error(f'Recieved unknown schema "{ds.content["schema"]["type"]}" type for dataset.')
                raise Exception(f'Recieved unknown schema "{ds.content["schema"]["type"]}" type for dataset.')
            
            logger.info(f'Found schema type "{ds.content["schema"]["type"]}" for dataset.')
        
        except KeyError:
            logger.error('Recieved dataset with no schema.')
            raise Exception('Recieved dataset with no schema.')


        if query_spark:
                if query is None:
                    query = f'SELECT * FROM {ds.id}'
                try:
                    query_result = ds.query_sourcedata(query)
                except Exception as e:
                    logger.error(f'Error querying data with SEDAR API: {e}')
                    raise Exception(f'Error querying data with SEDAR API: {e}')
                
                df = pd.DataFrame.from_dict(query_result['body'])
                logger.info("Structured data has been queried successfully.")
                return df
        
        elif query_raw:
                if not file_save_location:
                    raise ValueError('File save location must be provided for raw query.')
                logger.info(f"Attempting to save unstructured data.")

                if not os.path.exists(file_save_location):
                    logger.info(f"Creating directory {file_save_location}")
                    os.makedirs(file_save_location)

                ws_id = ws.content['id']
                source_id = ds.content['datasource']['id']
                
                try:
                    source_filenames = ds.content['datasource']['revisions'][-1]['source_files']
                except KeyError:
                    logger.error(f'Recieved dataset with no source files: {ds.content["datasource"]["revisions"][-1]["source_files"]} for revision {ds.content["datasource"]["revisions"][-1]}')
                    raise Exception(f'Recieved dataset with no source files: {ds.content["datasource"]["revisions"][-1]["source_files"]} for revision {ds.content["datasource"]["revisions"][-1]}')
                
                write_paths = []
                logger.info(f"Recieved source files: {source_filenames}")

                for file in source_filenames:
                    dest_path = os.path.join(file_save_location, file[5:])
                    hdfs_path = f"/datalake/{ws_id}/data/unstructured/{source_id}/{file[5:]}"
                    try:
                        data = self.hdfs.read_file(hdfs_path)
                    except Exception as e:
                        logger.error(f'Error reading file {hdfs_path} from HDFS: {e}')
                        raise Exception(f'Error reading file {hdfs_path} from HDFS: {e}')
                    
                    if not data:
                        logger.error(f"Recieved empty data for file {hdfs_path}")
                        raise Exception(f"Recieved empty data for file {hdfs_path}")
                    
                    logger.info(f"Saving orig. file {file} to {dest_path}.")
                    with open(dest_path, 'wb') as f:
                        f.write(data)

                    write_paths.append(dest_path)
                logger.info("Unstructured data has been saved successfully.")
                return write_paths

    #---------------------------------------------------------------------------------------------#
    def _convert_masks_for_gluon(self, mask_list: List[str], replace: bool = True) -> List[str]:
        """ convert masks to binary (PIL code L) and map to 0 and 1 values."""
        logger.info(f"Converting and binarizing all mask images.")

        def convert_and_binarize(read_path):            
            try:
                with Image.open(read_path) as img:
                    logger.debug(f"Got image with mode {img.mode}, size {img.size} and format {img.format}")
                    img = img.convert('L')
                    img_array = np.array(img)
                    binary_array = np.where(img_array > 0, 1, 0)
                    binary_img = Image.fromarray(binary_array.astype(np.uint8) * 255, 'L')
                    if replace:
                        dest = read_path
                    else:
                        dest = read_path + '_converted'
                    binary_img.save(dest, 'PNG')
                    logger.debug(f"Mask image {read_path} converted and saved to {dest}")
                    return dest
            except Exception as e:
                logger.error(f"Failed to convert and save mask image at {read_path}: {e}")
                raise Exception(f"Failed to convert and save mask image at {read_path}: {e}")
        
        new_masks = []
        for full_file in mask_list:
            if isImgFile(full_file):
                try:
                    replaced_file = convert_and_binarize(full_file)
                    new_masks.append(replaced_file)
                except Exception as e:
                    logger.error(f"Error processing file {full_file}: {e}")
                    raise ValueError(f"Error processing file {full_file}") from e
            else:
                logger.info(f"File {full_file} is not an image.")
                pass
        
        return new_masks
    
    # --------------------------------------------------------------------------------------------#
    def segmentation_as_data_frame(self, zip_path: str, unzip_path: str, convert_masks: bool = True) -> pd.DataFrame:
        
        try:
            file_paths = self.extract_zip(zip_path, unzip_path)
        except Exception as e:
            logger.error(f"Failed to extract zip file {zip_path}: {e}")
            raise

        image_list = []
        mask_list = []

        for path in file_paths:
            full_path = os.path.join(unzip_path, path)

            if (('images/' in path or 'image/' in path or 'data/' in path or 'img/' in path) and
            isImgFile(full_path)):
                image_list.append(full_path)
            elif (('mask/' in path or 'masks/' in path or 'label/' in path or 'labels/' in path) and
            isImgFile(full_path)):
                mask_list.append(full_path)
        
        image_list = sorted(image_list)
        mask_list = sorted(mask_list)

        if convert_masks:
            mask_list = self._convert_masks_for_gluon(mask_list)

        df = pd.DataFrame(data={'image': image_list, 'label': mask_list})
        return df

    # --------------------------------------------------------------------------------------------#
    def cocoAsDataFrame(self, zip_path, unzip_path):
        try:
            file_paths = self.extract_zip(zip_path, unzip_path)
        except Exception as e:
            logger.error(f"Failed to extract zip file {zip_path}: {e}")
            raise

        json_files = [path for path in file_paths if path.endswith('.json')]
        if not json_files:
            logger.error(f"No label files found in {unzip_path}")
            raise Exception(f"No label files found in {unzip_path}")
        if len(json_files) > 1:
            logger.info(f"More than one label file found in {unzip_path}: {json_files}. Using the first one.")

        label_file = [path for path in file_paths if path.endswith('.json')][0]        

        from autogluon.multimodal.utils.object_detection import from_coco

        label_file = os.path.join(unzip_path, label_file)
        root = os.path.dirname(label_file)
        logger.info(f"Attempting to load COCO annotations from {root}.")
        if os.path.basename(root) == 'annotations':
            root = os.path.dirname(root)
            logger.info(f"Failed. Attempting to load COCO annotations from {root}.")
        
        try:
            df = from_coco(label_file, root=root)
        except Exception as e:
            logger.error(f"Failed to load COCO annotations: {e}")
            raise Exception(f"Failed to load COCO annotations: {e}")

        return df

    # --------------------------------------------------------------------------------------------#
    def classificationAsDataFrame(self, zip_path, unzip_path):
        try:
            file_paths = self.extract_zip(zip_path, unzip_path)
        except Exception as e:
            logger.error(f"Failed to extract zip file {zip_path}: {e}")
            raise

        class_mapping = {}
        data = []
    
        for path in file_paths:
            
            directory, file_name = os.path.split(path)
            
            if directory and file_name:  # FILE INSIDE FOLDER
                if directory not in class_mapping:
                    class_mapping[directory] = len(class_mapping)
                
                full_path = os.path.join(unzip_path, path)
                class_id = class_mapping[directory]
                data.append([full_path, class_id])
    
        df = pd.DataFrame(data, columns=['image', 'label'])
        return df, class_mapping
    
    #--------------------------------------------------------------------------------------------#
    @staticmethod
    def imageDataFrameToNumpyXy(df: pd.DataFrame, target: str = 'label', task_type: str = 'classification') -> Tuple[np.ndarray, np.ndarray]:
        images = []
        labels = []

        for index, row in df.iterrows():
            try:
                with Image.open(row['image']) as img:                  
                    logger.debug(f"Got image with mode {img.mode}, size {img.size} and format {img.format}")
                    if img.mode in ['L', '1'] and len(img_arr.shape) == 2: 
                        img_arr = np.array(img)
                        img_arr = img_arr.reshape(img_arr.shape[0], img_arr.shape[1], 1) 
                    if img.mode == 'RGBA':
                        img = img.convert('RGB')
                        img_arr = np.array(img)
                    if img.mode == 'RGB':
                        img_arr = np.array(img)
                        try:
                            if img_arr.shape[2] != 3:
                                img_arr = img_arr.reshape(img_arr.shape[0], img_arr.shape[1], 3)
                        except Exception as e:
                            pass
                    else:
                        print(f'unhandled image mode {img.mode}')
                        img_arr = np.array(img)
                    logger.debug(f"Converted image to array with shape {img_arr.shape}")

                    images.append(img_arr)
                    
                    if isTaskClassification(task_type):
                        labels.append(row[target])
                    elif isTaskSegmentation(task_type):
                        raise NotImplementedError('Segmentation not implemented yet.')
                    elif isTaskObjectDetection(task_type):
                        raise NotImplementedError('Object detection not implemented yet.')
                    else:
                        raise Exception(f'Unknown type {task_type} for image data.')

            except IOError:
                print(f"Could not read image: {row['image']}")
        
        X = np.array(images)
        y = np.array(labels)
        return X, y
    
    # --------------------------------------------------------------------------------------------#
    def segmentationAsNumpyXy(self, zip_path, unzip_path):
        file_paths = self.extract_zip(zip_path, unzip_path)

        image_paths = []
        mask_paths = []

        for path in file_paths:
            base_name = os.path.basename(path)
            name, ext = os.path.splitext(base_name)
            if ext not in ['.jpg', '.png']:
                continue

            if 'images/' in path or 'image/' in path or 'data/' in path or 'img/' in path:
                image_paths.append((name, os.path.join(unzip_path, path)))
            elif 'mask/' in path or 'masks/' in path or 'label/' in path or 'labels/' in path:
                mask_paths.append((name, os.path.join(unzip_path, path)))

        image_paths.sort(key=lambda x: x[0])
        mask_paths.sort(key=lambda x: x[0])

        X, Y = [], []
        for (img_name, img_path), (mask_name, mask_path) in zip(image_paths, mask_paths):
            if img_name == mask_name: 
                with Image.open(img_path) as img:
                    img_arr = np.array(img)
                    if img.mode != 'RGB':
                        img_arr = img_arr.reshape(img_arr.shape[0], img_arr.shape[1], 1) 
                    X.append(img_arr)

                with Image.open(mask_path) as mask:
                    mask_arr = np.array(mask.convert('1'))
                    if mask.mode != 'RGB':
                        mask_arr = mask_arr.reshape(mask_arr.shape[0], mask_arr.shape[1], 1)
                    Y.append(mask_arr)
            else:
                print(f"No matching file for {img_name} and {mask_name}")

        return np.array(X), np.array(Y)


    # --------------------------------------------------------------------------------------------#
    @staticmethod
    def split_train_test(data, label=None, train_size=0.8):
        if data is None:
            return None, None
        
        if isinstance(data, pd.DataFrame):
            return train_test_split(data, train_size=train_size)
        
        if isinstance(data, np.ndarray):
            if not label:
                raise Exception('label must be provided for numpy array.')
            return train_test_split(data, label, train_size=train_size)
        
    # --------------------------------------------------------------------------------------------#
    def clean_dir(dir_path):
        if os.path.exists(dir_path):
            shutil.rmtree(dir_path)
        os.makedirs(dir_path)

    # --------------------------------------------------------------------------------------------#
    def extract_zip(self, zip_path, unzip_path):

        if not zip_path.endswith('.zip'):
            zip_files = [f for f in os.listdir(zip_path) if f.endswith('.zip')]
            if not zip_files:
                raise Exception(f'No zip files found in {zip_path}')
            zip_path = os.path.join(zip_path, zip_files[0])

        if not os.path.exists(unzip_path):
            os.makedirs(unzip_path)

        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(unzip_path)
            file_names = zip_ref.namelist()
        
        return file_names
    

    # --------------------------------------------------------------------------------------------#
    def CAAFEFeatureEngineering(self, workspace_id, dataset_id, file_save_location, data_type, 
                                task_type, target=None, dataset_description="", model="gpt-3.5-turbo",
                                iterations = 3):
        
        from caafe import CAAFEImageClassifier, CAAFEClassifier, CAAFEImageSegmentor
        from sklearn.metrics import accuracy_score, jaccard_score
        from sklearn.ensemble import RandomForestClassifier
        from tabpfn.scripts import tabular_metrics
        import openai

        if isDataTypeTabular(data_type) and isTaskClassification(task_type):
            df = self.query_data(workspace_id, dataset_id)
            df_train, df_test = self.split_train_test(df)


            base_clf = RandomForestClassifier()
            base_clf.fit(df_train.drop(columns=[target]), df_train[target])
            pred = base_clf.predict(df_test.drop(columns=[target]))
            acc = accuracy_score(pred, df_test[target])
            print(f'Accuracy before CAAFE {acc}')

            caafe_clf = CAAFEClassifier(
                base_classifier=base_clf,
                llm_model=model,
                iterations=iterations
                )

            if not target:
                raise Exception('target must be provided for tabular data.')
            
            caafe_clf.fit_pandas(
                df_train,
                target_column_name=target,
                dataset_description=dataset_description
                )

            pred = caafe_clf.predict(df_test)
            acc = accuracy_score(pred, df_test[target])
            print(f'Accuracy after CAAFE {acc}')

            return caafe_clf.apply_code(df)

        elif isDataTypeImage(data_type) and isTaskClassification(task_type): 
            loc = self.query_data(workspace_id, dataset_id, file_save_location=file_save_location)
            X, y, map = self.zip_to_class_np(loc[0], file_save_location+'_unzipped')

            caafe_clf = CAAFEImageClassifier(
                llm_model=model,
                iterations=iterations
                )

            pred = caafe_clf.performance_before_run(X, y)
            acc = accuracy_score(pred, y)
            print(f'Accuracy before CAAFE {acc}')    
            caafe_clf.fit_images(
                X,
                y,
                dataset_description
                )

            pred = caafe_clf.predict(X)
            acc = accuracy_score(pred, y)
            print(f'Accuracy after CAAFE {acc}')
            X_, y_ = caafe_clf.apply_code(X, y)

            return X_, y_, map
        
        elif isDataTypeImage(data_type) and isTaskSegmentation(task_type):
            loc = self.query_data(workspace_id, dataset_id, file_save_location=file_save_location)
            X, y = self.segmentationAsNumpyXy(loc[0], file_save_location+'_unzipped')

            caafe_clf = CAAFEImageSegmentor(
                llm_model=model,
                iterations=iterations
                )

            pred = caafe_clf.performance_before_run(X, y)
            if len(np.unique(y)) == 2:
                iou = iou = jaccard_score(y.flatten(), pred.flatten(), average='binary')
            else:
                iou = iou = jaccard_score(y.flatten(), pred.flatten(), average='macro')

            print(f'IoU before CAAFE {iou}') 


            caafe_clf.fit_images(
                X,
                y,
                dataset_description
                )

            pred = caafe_clf.predict(X)
            if len(np.unique(y)) == 2:
                iou = iou = jaccard_score(y.flatten(), pred.flatten(), average='binary')
            else:
                iou = iou = jaccard_score(y.flatten(), pred.flatten(), average='macro')

            print(f'IoU after CAAFE {iou}') 
            X_, y_ = caafe_clf.apply_code(X, y)
            return X_, y_,

        else:
            raise Exception('Not implemented yet.')
                                    


