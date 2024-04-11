import io
import os

import numpy as np
import pandas as pd

from PIL import Image

try:
    from keras.datasets import mnist
except ImportError:
        print("WARNING: Keras not installed. Some functionalities might not be available.")
        mnist = None
try:
    from autogluon.multimodal.utils.misc import shopee_dataset
    from autogluon.core.utils.loaders import load_zip
    from autogluon.multimodal.utils.object_detection import from_coco
    from autogluon.core.utils.loaders import load_pd
except ImportError:
    print("WARNING: AutoGluon not installed. Some functionalities might not be available.")
    shopee_dataset = None
    load_zip = None
    from_coco = None
    load_pd = None


def create_leaf_df(n_samples: int = 2000):
    """
        dtype: image
        problem: semantic segmentation
    """
    download_dir = './data/ag_automm_tutorial'
    zip_file = 'https://automl-mm-bench.s3.amazonaws.com/semantic_segmentation/leaf_disease_segmentation.zip'
    load_zip.unzip(zip_file, unzip_dir=download_dir)

    dataset_path = os.path.join(download_dir, 'leaf_disease_segmentation')
    train_data = pd.read_csv(f'{dataset_path}/train.csv', index_col=0)
    val_data = pd.read_csv(f'{dataset_path}/val.csv', index_col=0)
    test_data = pd.read_csv(f'{dataset_path}/test.csv', index_col=0)

    image_col = 'image'
    label_col = 'label'

    def path_expander(path, base_folder):
        path_l = path.split(';')
        return ';'.join([os.path.abspath(os.path.join(base_folder, path)) for path in path_l])

    for per_col in [image_col, label_col]:
        train_data[per_col] = train_data[per_col].apply(lambda ele: path_expander(ele, base_folder=dataset_path))
        val_data[per_col] = val_data[per_col].apply(lambda ele: path_expander(ele, base_folder=dataset_path))
        test_data[per_col] = test_data[per_col].apply(lambda ele: path_expander(ele, base_folder=dataset_path))

    return train_data[:n_samples]


def create_sentiment_treebank_df(n_samples: int = 2000):
    """
        dtype: text
        problem: binary classification        
    """
    train_data = load_pd.load('https://autogluon-text.s3-accelerate.amazonaws.com/glue/sst/train.parquet')
    test_data = load_pd.load('https://autogluon-text.s3-accelerate.amazonaws.com/glue/sst/dev.parquet')

    return train_data[:n_samples]

def create_mloc_df(n_samples: int = 2000):
    """
        dtype: text
        problem: multiclass classification        
    """
    download_dir = "./data/ag_automm_tutorial_fs_cls"
    zip_file = "https://automl-mm-bench.s3.amazonaws.com/nlp_datasets/MLDoc-10shot-en.zip"
    load_zip.unzip(zip_file, unzip_dir=download_dir)
    dataset_path = os.path.join(download_dir)
    train_df = pd.read_csv(f"{dataset_path}/train.csv", names=["label", "text"])
    test_df = pd.read_csv(f"{dataset_path}/test.csv", names=["label", "text"])
    
    return train_df[:n_samples]

def create_shopee_df(is_bytearray = False, n_samples: int = 2000):
    """
        dtype: image
        problem: multiclass classification
    """
    download_dir = './data/shopee'
    shopee_train, shopee_test = shopee_dataset(download_dir, is_bytearray=is_bytearray)

    return shopee_train[:n_samples]

def create_coco_motorbike_df(n_samples: int = 2000):
    """
        dtype: image
        problem: object detection
    """
    zip_file = "https://automl-mm-bench.s3.amazonaws.com/object_detection_dataset/tiny_motorbike_coco.zip"
    download_dir = "./data/tiny_motorbike_coco"

    load_zip.unzip(zip_file, unzip_dir=download_dir)
    data_dir = os.path.join(download_dir, "tiny_motorbike")
    train_path = os.path.join(data_dir, "Annotations", "trainval_cocoformat.json")
    test_path = os.path.join(data_dir, "Annotations", "test_cocoformat.json")

    train_df = from_coco(train_path)
    return train_df[:n_samples]


def __save_jpg_format(img):
    img_2d = img.reshape(int(np.sqrt(img.shape[0])), -1)
    img_pil = Image.fromarray(img_2d)
    buffer = io.BytesIO()
    img_pil.save(buffer, format='JPEG')
    img_bytes = bytearray(buffer.getvalue())
    return img_bytes

def create_mnist_bytearray_df(label_col: str = 'label', n_samples: int = 500):
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    flattened_images = []
    for img in x_train.reshape(x_train.shape[0], -1):
        img_bytes = __save_jpg_format(img)
        flattened_images.append([img_bytes])
    
    return pd.DataFrame(list(zip(flattened_images, y_train)), columns=['image', label_col])[:n_samples]

def create_mnist_tuple(n_samples: int = 500):
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x = np.concatenate((x_train, x_test))
    y = np.concatenate((y_train, y_test))

    #y = y.reshape(-1, 1)
    #data = np.hstack((x.reshape(x.shape[0], -1), y))
    #return data[:n_samples]
    # y = y[:, np.newaxis, np.newaxis]
    # data = np.concatenate((x, y), axis=1)
    # return data[:n_samples]
    return x[:n_samples], y[:n_samples]

def create_glass_df():
    path = os.path.join(os.path.dirname(__file__), './data/glass.csv')
    if not os.path.exists(path):
        path = os.path.join(os.path.dirname(__file__), '../data/glass.csv')
        if not os.path.exists(path):
            path = os.path.join(os.path.dirname(__file__), '../../data/glass.csv')
    return pd.read_csv(path)

def create_m4_df(n_samples: int = 2000):
    return pd.read_csv("https://autogluon.s3.amazonaws.com/datasets/timeseries/m4_hourly_subset/train.csv")[:n_samples]