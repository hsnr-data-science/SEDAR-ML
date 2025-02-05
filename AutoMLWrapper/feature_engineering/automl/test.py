#!/usr/bin/env python
# coding: utf-8

# # TODO
# #### Klassen protecten
# ```
# import AutoMLWrapper.automlwrapper as aw
# aw.Configuration.Configuration
# ```
# 
# #### Erweiterung um Bibliotheken
# -  @__library.setter
#     - a) Global/GlobalConfig.yaml
#         - lib-names
#         - type mappings
#     - b) directory durchsuchen 
# - import in AutoMLWrapper anpassen
# - SEDAR automl.py config locations anpassen
# 
# ### custom proprocessing
# - typen checken
# - die wrapper könnten property methods implementieren, die die Rückgabe eines festen samples prüfen
# 
# ### custom parameters
# 
# 
# ### 2+ problem types

# In[1]:


import io
import mlflow
import pandas as pd
import numpy as np
from PIL import Image

from AutoMLWrapper.tests import DataSamples

#from automlwrapper import AutoMLWrapper
from AutoMLWrapper.automlwrapper import AutoMLWrapper, SedarDataLoader



# In[2]:


RUN_ALL = 0
UPLOAD_ALL = 0

RUN_KERAS = 0
UPLOAD_KERAS = 0

RUN_SKLEARN = 0
UPLOAD_SKLEARN = 0

RUN_GLUON = 1
UPLOAD_GLUON = 0

user_hp = {'epochs':20, 'time_limit': 360, 'memory_limit':7000, 'testtest': 123, 'num_trials': 4,}

# In[3]:


MLFLOW = 0

if MLFLOW:
    
    # #experiment_id = mlflow.create_experiment('test_automl', artifact_location='/home/mibbels/sedar-masterarbeit/sedar/sedar/automl/tmp/artifacts')
    
    experiment_id = 5
    remote_server_uri = "http://127.0.0.1:6798"
    mlflow.set_tracking_uri(remote_server_uri) 
    mlflow.set_experiment(experiment_id=experiment_id)

# # Daten

# In[4]:


""" Bilddaten """
#mnist_byte_df = DataSamples.create_mnist_bytearray_df(label_col='label', n_samples=-1)
#shopee_path_df = DataSamples.create_shopee_df(is_bytearray = False, n_samples=-1)
#mnist_tp = DataSamples.create_mnist_tuple(n_samples=-1)
#leaf_df = DataSamples.create_leaf_df(n_samples=-1)

#coco_df = DataSamples.create_coco_motorbike_df(n_samples=-1)

""" Textdaten """
#sentiment_df = DataSamples.create_sentiment_treebank_df(n_samples=-1)
#mloc_df = DataSamples.create_mloc_df(n_samples=-1)

""" Tabellendaten """
glass_df = DataSamples.create_glass_df()

""" Timeseries """
#m4_df = DataSamples.create_m4_df(n_samples=-1)

# In[5]:


ask = AutoMLWrapper('autosklearn')
ask.SetOutputDirectory('test/ask/2')

if RUN_ALL or RUN_SKLEARN:
    ask.Train(
        data=glass_df,
        target_column='Type',
        task_type='classification',
        data_type='tabular',
        problem_type='multiclass',
        hyperparameters=user_hp
    )


if UPLOAD_ALL or UPLOAD_SKLEARN:
    print(ask.Output())
    ask.MlflowUploadBest({})

# In[8]:


agl = AutoMLWrapper('autogluon')
agl.SetOutputDirectory('test/agl/9')


if RUN_ALL or RUN_GLUON: 
    train, test = SedarDataLoader.split_train_test(glass_df)
    agl.Train(
        data=train,
        target_column='Type',
        data_type='tabular',
        task_type='classification',        
        problem_type='multiclass',
        hyperparameters=user_hp
    )
    ev = agl.Evaluate(test)
    
if UPLOAD_ALL or UPLOAD_GLUON:
    print(agl.Output())
    agl.MlflowUploadBest({})

# In[7]:


print(ev)


# In[ ]:


akr = AutoMLWrapper('autokeras')
akr.SetOutputDirectory('test/akr/1')

if RUN_ALL or RUN_KERAS:
    akr.Train(data=mnist_tp,
        target_column='Type',
        task_type='classification',
        data_type='image',
        problem_type='multiclass',
        hyperparameters=user_hp
    )

if UPLOAD_ALL or UPLOAD_KERAS:
    print(akr.Output())
    akr.MlflowUploadBest({})

# # MLFlow
# ~/miniconda3/envs/automl_env/bin/pip

# In[ ]:


if MLFLOW:
    df = df_glass.drop(columns=['Type'])
    logged_model = 'runs:/a0337b9ecfe84e99b59ff3a9639f1ef6/model'
    loaded_model = mlflow.pyfunc.load_model(logged_model)

    #loaded_model.predict(test_data_byte)

# # SEDAR API

# In[ ]:




# # DataLoader TODO
# - sedar api raw sourcedata
# - sedar api get_all_images (image metadata)
# 
# - query_sourcedata nur structured
#     - image
#     - unstructured
#     - semistructured
# 
# - keras format
# 
# # Ingestion
# - metadaten?

# In[ ]:


from sedarapi import SedarAPI
from AutoMLWrapper.automlwrapper.SedarDataLoader import SedarDataLoader

# In[1]:


tabcl = ["13b4787c3e454649aa05a4cd680edc37", "986f2e837ca44f3e8c0ee7d2dc0c4287",
          "./data/sedar_raw/zip", "tabular", "classification", "Type"]
imseg = ["13b4787c3e454649aa05a4cd680edc37", "324ea420125d4167a76151b62368c4ad",
          "./data/sedar_raw/zip", "image", "segmentation", None]
imcl = ["13b4787c3e454649aa05a4cd680edc37", "513c6b1ee46b478c8e0925a098d2f387",
          "./data/sedar_raw/zip", "image", "classification", None]

from AutoMLWrapper.automlwrapper.SedarDataLoader import SedarDataLoader
from sedarapi import SedarAPI


def test_sedar_caafe(workspace_id, dataset_id, file_save_location, data_type, problem_type, target):
    base_url = "http://192.168.220.107:5000"
    email = "admin"
    password = "admin"
    sedar = SedarAPI(base_url)
    sedar.connection.logger.setLevel("INFO")
    sedar.login(email, password)

    DataLoader = SedarDataLoader(sedar)
    DataLoader.CAAFEFeatureEngineering(
        workspace_id, 
        dataset_id, 
        file_save_location,
        data_type,
        problem_type,
        target,
        dataset_description='No description',
        iterations=1
    )
    
test_sedar_caafe(*imseg)

# In[ ]:


base_url = "http://192.168.220.107:5000"
email = "admin"
password = "admin"

sedar = SedarAPI(base_url)
sedar.connection.logger.setLevel("INFO")
sedar.login(email, password)

# In[ ]:


workspace_id = '13b4787c3e454649aa05a4cd680edc37'
dataset_id = '324ea420125d4167a76151b62368c4ad'

# 986f2e837ca44f3e8c0ee7d2dc0c4287  - glass
# b5b74391e41e4634a54d5cffa059663b - coco_22_train_v1 
# 324ea420125d4167a76151b62368c4ad - gitter_train_v1
# 513c6b1ee46b478c8e0925a098d2f387  - test classification 1 img each class

DataLoader = SedarDataLoader(sedar)
#loc = DataLoader.query_data(workspace_id, dataset_id, file_save_location='./data/sedar_raw/test')
#X, y, mapping = DataLoader.zip_to_class_np_keras(loc[0], './data/sedar_raw/test/test_coco')


# In[ ]:


from tabpfn.scripts import tabular_metrics

if 0:
    DataLoader.CAAFEFeatureEngineering(
        workspace_id, 
        '986f2e837ca44f3e8c0ee7d2dc0c4287', 
        file_save_location='./data/sedar_raw/test',
        data_type='tabular',
        problem_type='classification',
        target='Type',
        dataset_description='No description',
        iterations=1
    )

# In[ ]:


DataLoader.CAAFEFeatureEngineering(
    workspace_id, 
    '324ea420125d4167a76151b62368c4ad', 
    file_save_location='./data/sedar_raw/test',
    data_type='image',
    problem_type='segmentation',
    dataset_description='No description',
)

# In[ ]:




# In[ ]:


X, y = DataLoader.zip_to_segmentation_np_keras(loc[0], './data/sedar_raw/test/test_seg')

# In[ ]:


from CAAFE.caafe import CAAFEImageClassifier, CAAFEImageSegmentor
from sklearn.metrics import accuracy_score
import openai

# In[ ]:


if 1:
    import tensorflow as tf
    import numpy as np
    import keras

    cifar10 = keras.datasets.cifar10
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()
    X_train, X_test = X_train / 255.0, X_test / 255.0

    # If you want to use the entire dataset as X and y
    X = np.concatenate((X_train, X_test), axis=0, dtype=np.float32)
    y = np.concatenate((y_train, y_test), axis=0, dtype=np.float32)

    #X = X.reshape(X.shape[0], 28, 28, 1)
    X = X[:1000]
    y = y[:1000]


# In[ ]:


#model = "gpt-3.5-turbo"
model = "gpt-4"
caafe_clf = CAAFEImageClassifier(llm_model=model,
                                 iterations=3,
                                 )

pred = caafe_clf.performance_before_run(X, y)
acc = accuracy_score(pred, y)
print(f'Accuracy before CAAFE {acc}')


caafe_clf.fit_images(
    X,
    y,
    dataset_description="This is the CIFAR10 dataset."
    )

pred = caafe_clf.predict(X)
acc = accuracy_score(pred, y)
print(f'Accuracy after CAAFE {acc}')

X, y = caafe_clf.apply_code(X, y)

# In[ ]:


_, y = caafe_clf.apply_code(X, y)
pred = caafe_clf.predict(X)
acc = accuracy_score(pred, y)
print(f'Accuracy after CAAFE {acc}')



# In[ ]:


if 0:
    ws = sedar.get_workspace('13b4787c3e454649aa05a4cd680edc37')
    dataset = ws.get_dataset('986f2e837ca44f3e8c0ee7d2dc0c4287')
    
    # try:
    #     success = dataset.delete()
    #     if success:
    #         print("Dataset deleted successfully.")
    # except Exception as e:
    #     print(e)


# In[ ]:


!cd ./AutoMLWrapper/tests && python3 -m unittest TestAutoMLWrapper.py

