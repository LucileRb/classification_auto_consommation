######## IMPORTS ########

# sklearn
from sklearn import cluster, metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, auc, roc_auc_score, roc_curve
from sklearn import preprocessing
from sklearn import manifold, decomposition

from plot_keras_history import show_history, plot_history


# tensorflow
import tensorflow as tf
from keras.models import Model, Sequential
#from tensorflow.keras.optimizers.legacy import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import GlobalAveragePooling2D, GlobalAveragePooling1D, Flatten, Dense, Dropout
from keras.layers import Rescaling, RandomFlip, RandomRotation, RandomZoom
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
from keras.utils import load_img, img_to_array
from keras.utils import to_categorical



##########################################################################################################################################################################
################################################## Fonctions de création de modèles images blabla ######################################################################
# DIRE A QUOI CA SERT
def create_model_fct_1():
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

def create_model_fct_2():
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
    model.compile(loss = 'sparse_categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy']) # optimizer = 'rmsprop'

    print(model.summary())

    return model

# DIRE A QUOI CA SERT -> DATA AUGMENTATION
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