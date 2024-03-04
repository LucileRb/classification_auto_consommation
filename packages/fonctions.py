########## Imports ##########

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
from IPython.display import display
from sklearn import cluster, metrics

# sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, auc, roc_auc_score, roc_curve
from sklearn import preprocessing
from sklearn import manifold, decomposition
from sklearn import cluster, metrics

from plot_keras_history import show_history, plot_history


# tensorflow
import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
#from tensorflow.keras.optimizers.legacy import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import GlobalAveragePooling2D, GlobalAveragePooling1D, Flatten, Dense, Dropout
from tensorflow.keras.layers import Rescaling, RandomFlip, RandomRotation, RandomZoom
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.utils import to_categorical

########## Fonctions ##########

# Labels sur graphs
def addlabels(x, y):
    """ Fonction pour ajouter valeurs sur graphs """
    for i in range(len(x)):
        plt.text(i, y[i]//2, y[i], ha = 'center', fontstyle = 'italic')


# Duplicats
def remove_duplicates(df):
    """Fonction pour détecter les doublons dans un jeu de données et les supprimer si il y en a"""

    print('********** Détection des doublons **********\n')
    
    # Nombre de duplicats dans le jeu de données
    doublons = df.duplicated().sum()
    print(f'Nombre de duplicats dans le jeu de données = {doublons}')

    if doublons > 0:

        # Affichier le pourcentage de duplicats
        print(f"\nPourcentage de duplicats : {round((df.duplicated().sum().sum()/np.product(df.shape))*100, 2)}\n")

        # supprimer duplicats
        print('****** Suppression des duplicats en cours ******')
        df.drop_duplicates(inplace = True)

        # nombre de duplicats dans le jeu de données après processing
        print(f'Nombre de duplicats dans le jeu de données après processing: {df.duplicated().sum()}')



# Données manquantes
def nan_detection(df):
    """Fonction pour détecter les données manquantes dans un jeu de données et afficher insights pertinents"""

    print('********** Détection des données manquantes **********\n')

    # Nombre total de nan :
    total_nan = df.isna().sum().sum()
    print(f'Nombre de données manquantes dans le jeu de données = {total_nan}')

    if total_nan > 0:
    
        # Pourcentage
        print(f"\nPourcentage de valeurs manquantes : {round((df.isna().sum().sum()/np.product(df.shape))*100, 2)}\n")

        # Nan et pourcentage de nan par features
        print('\nValeurs manquantes par colonne : \n')
        pd.set_option('display.max_rows', None) # pour afficher toutes les lignes
        values = df.isnull().sum()
        percentage = 100 * values / len(df)
        table = pd.concat([values, percentage.round(2)], axis = 1)
        table.columns = ['Nombres de valeurs manquantes', '% de valeurs manquantes']
        display(table[table['Nombres de valeurs manquantes'] != 0].sort_values('% de valeurs manquantes', ascending = False))
        pd.reset_option('display.max_rows') # on reset l'option pour ne plus afficher toutes les lignes

        # Heatmap
        print('\nHeatmap des valeurs manquantes : \n')
        plt.figure(figsize = (15, 7))
        sns.heatmap(df.isna(), cbar = False)
        plt.show()


########## Catégorie ##########
def get_category(row):
    match = re.search(r'(?:\w+\s+){2}(?=\>\>)', row['product_category_tree'])
    if match:
        result = match.group(0).strip()
        return result



def data_fct(df, path):
    list_photos = [file for file in path]
    print(len(list_photos))
    data = pd.DataFrame()
    data['image_path'] = list_photos
    data['label_name'] = df['category']
    return data

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


def create_model_fct():
    # Récupération modèle pré-entraîné
    model0 = VGG16(include_top = False, weights = 'imagenet', input_shape = (224, 224, 3))

    # Layer non entraînables = on garde les poids du modèle pré-entraîné
    for layer in model0.layers:
        layer.trainable = False

    # Récupérer la sortie de ce réseau
    x = model0.output
    # Compléter le modèle
    x = GlobalAveragePooling2D()(x)
    x = Dense(256, activation = 'relu')(x)
    x = Dropout(0.5)(x)
    predictions = Dense(4, activation = 'softmax')(x) # 4 = nb de catégories

    # Définir le nouveau modèle
    model = Model(inputs = model0.input, outputs = predictions)
    # compilation du modèle 
    model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy']) # optimizer = 'rmsprop'

    print(model.summary())

    return model

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

def create_model_fct2():
    # Data augmentation
    data_augmentation = Sequential([
        RandomFlip('horizontal', input_shape = (224, 224, 3)),
        RandomRotation(0.1),
        RandomZoom(0.1),
        # Rescaling(1./127.5, offset=-1.0)
      ])

    # Récupération modèle pré-entraîné
    model_base = VGG16(include_top = False, weights = 'imagenet', input_shape = (224, 224, 3))
    for layer in model_base.layers:
        layer.trainable = False

    # Définition du nouveau modèle
    model = Sequential([
                data_augmentation,
                Rescaling(1./127.5, offset = -1),
                model_base,
                GlobalAveragePooling2D(),
                Dense(256, activation = 'relu'),
                Dropout(0.5),
                Dense(4, activation = 'softmax')
                ])

    # compilation du modèle 
    model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])

    print(model.summary())

    return model

#def resize_and_rescale(image, label):
#    image = tf.cast(image, tf.float32)
#    image = tf.image.resize(image, [IMG_SIZE, IMG_SIZE])
#    image = (image / 255.0)
#    return image, label


# LIRE LA DOC
def dataset_fct(batch_size, path, validation_split = 0, data_type = None):
    dataset = tf.keras.utils.image_dataset_from_directory(
                    path,
                    labels = None, # changer 'inferred' en None (cf doc)
                    label_mode = 'categorical',
                    class_names = None, batch_size = batch_size, image_size = (224, 224), shuffle = True, seed = 42,
                    validation_split = validation_split, subset = data_type
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
