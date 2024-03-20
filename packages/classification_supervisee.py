######## IMPORTS ########

# keras
from keras.models import Model, Sequential
from keras.layers import GlobalAveragePooling2D, Dense, Dropout, Rescaling, RandomFlip, RandomRotation, RandomZoom
from keras.applications.vgg16 import VGG16

##########################################################################################################################################################################
################################################## Fonctions de création de modèles images blabla ######################################################################

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

# DATA AUGMENTATION
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