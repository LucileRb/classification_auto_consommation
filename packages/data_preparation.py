######## IMPORTS ########
import numpy as np
import pandas as pd

# sklearn
from sklearn import metrics

# tensorflow
import tensorflow as tf

# Keras
from keras.applications.vgg16 import preprocess_input
from keras.utils import load_img, img_to_array


##########################################################################################################################################################################
################################################## Fonctions de préparation des données images ######################################################################

def list_fct(data, name):
    list_image_name = [data['image_path'][i] for i in range(len(data)) if data['label_name'][i] == name]
    return list_image_name

def conf_mat_transform(y_true, y_pred):
    conf_mat = metrics.confusion_matrix(y_true, y_pred)
    corresp = np.argmax(conf_mat, axis = 0)
    print ('Correspondance des clusters : ', corresp)
    # y_pred_transform = np.apply_along_axis(correspond_fct, 1, y_pred)
    labels = pd.Series(y_true, name = 'y_true').to_frame()
    labels['y_pred'] = y_pred
    labels['y_pred_transform'] = labels['y_pred'].apply(lambda x : corresp[x])

    return labels['y_pred_transform']


def image_prep_fct(data):
    prepared_images = []
    for image_num in range(len(data['image_path'])):
        img = (load_img(data['image_path'][image_num], target_size = (224, 224)))
        img = img_to_array(img)
        img = img.reshape((img.shape[0], img.shape[1], img.shape[2]))
        img = preprocess_input(img)
        prepared_images.append(img)
        prepared_images_np = np.array(prepared_images)
    return prepared_images_np


def dataset_fct(batch_size, path, labels = None, validation_split = 0, data_type = None):
    dataset = tf.keras.utils.image_dataset_from_directory(
                    path,
                    labels = labels,
                    label_mode = 'int',
                    class_names = None,
                    batch_size = batch_size,
                    image_size = (224, 224),
                    shuffle = True,
                    seed = 42,
                    validation_split = validation_split,
                    subset = data_type
                    )
    return dataset

def data_flow_fct(batch_size, data, datagen, data_type = None):
    data_flow = datagen.flow_from_dataframe(
        data,
        directory = '',
        x_col = 'image_path',
        y_col = 'label_name',
        weight_col = None,
        target_size = (256, 256),
        classes = None,
        class_mode = 'categorical',
        batch_size = batch_size,
        shuffle = True,
        seed = 42,
        subset = data_type
        )
    return data_flow