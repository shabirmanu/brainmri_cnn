import numpy as np
import pandas as pd
import tensorflow as tf
from scipy import misc, ndimage
import keras
from keras import backend as K
from keras.models import Sequential, load_model
from keras.layers import Activation
from keras.layers.core import Dense, Flatten
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import *
from keras.preprocessing.image import ImageDataGenerator,array_to_img, img_to_array, load_img
from keras.optimizers import Adam
from keras.metrics import categorical_crossentropy
from sklearn.metrics import confusion_matrix
import itertools
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import os
import matplotlib.image as mpimg


train_path = "Data/train"
valid_path = "Data/valid"
test_path = "Data/test"

IMG_W = 150
IMG_H = 150

train_datagen = ImageDataGenerator()

train_batches = train_datagen.flow_from_directory(train_path, target_size=(IMG_W, IMG_H),
                                                  classes=['normal', 'abnormal'], batch_size=5)
valid_batches = train_datagen.flow_from_directory(valid_path, target_size=(IMG_W, IMG_H),
                                                  classes=['normal', 'abnormal'], batch_size=2)
test_batches = train_datagen.flow_from_directory(test_path, target_size=(IMG_W, IMG_H), classes=['normal', 'abnormal'],
                                                 batch_size=6)

def train_model():

    '''Create a simple CNN'''
    model = Sequential([
        Conv2D(64, (4, 4), input_shape=(IMG_W, IMG_W, 3), activation='relu'),
        Flatten(),
        Dense(2, activation="softmax")
    ])

    model.compile(Adam(lr=.0001), loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit_generator(train_batches, steps_per_epoch=12, validation_data=valid_batches, validation_steps=3, epochs=5,
                        verbose=2)
    return model

def augment_images(img_path, amount, save_dir):
    datagen = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')

    img = load_img(img_path)  # this is a PIL image
    x = img_to_array(img)  # this is a Numpy array with shape (3, 150, 150)
    x = x.reshape((1,) + x.shape)  # this is a Numpy array with shape (1, 3, 150, 150)

    # aug_iter = gen.flow(image)
    i = 0
    for batch in datagen.flow(x, batch_size=1,
                              save_to_dir=save_dir, save_prefix='ab', save_format='jpeg'):
        i += 1
        if i > amount:
            break  # otherwise the generator would loop indefinitely


def train_with_vgg():
    vgg16_model = keras.applications.vgg16.VGG16()
    model = Sequential()
    i = 0
    total_layers = len(vgg16_model.layers) - 1
    print(total_layers)
    for layer in vgg16_model.layers:
        if (i < total_layers):
            model.add(layer)
        i += 1
    model.add(Dense(2, activation='softmax'))
    model.compile(Adam(lr=.0001), loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit_generator(train_batches, steps_per_epoch=12, validation_data=valid_batches, validation_steps=3, epochs=10,
                        verbose=2)
    return model


def test_model(model):
    test_imgs, test_labels = next(test_batches)
    plots(test_imgs, titles=test_labels)
    test_labels = test_labels[:, 0]

    predictions = model.predict_generator(test_batches, steps=1, verbose=2)

    print(predictions)


def plots(ims, figsize=(12, 6), rows=1, interp=False, titles=None):
    if type(ims[0]) is np.ndarray:
        ims = np.array(ims).astype(np.uint8)
        if (ims.shape[-1] != 3):
            ims = ims.transpose((0, 2, 3, 1))

    f = plt.figure(figsize=figsize)
    cols = len(ims) // rows if len(ims) % 2 == 0 else len(ims) // rows + 1
    for i in range(len(ims)):
        sp = f.add_subplot(rows, cols, i + 1)
        sp.axis('Off')
        if titles is not None:
            sp.set_title(titles[i], fontsize=16)
        plt.imshow(ims[i], interpolation=None if interp else 'none')


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

def custom_model():
    cmodel = Sequential()
    cmodel.add(Conv2D(32, (3, 3), input_shape=(150, 150, 3), activation='relu'))
    cmodel.add(MaxPooling2D(pool_size=(2, 2), strides=None, padding='valid', data_format='channels_last'))

    cmodel.add(Conv2D(32, (3, 3), activation='relu'))
    cmodel.add(MaxPooling2D(pool_size=(2, 2), strides=None, padding='valid', data_format='channels_last'))

    cmodel.add(Conv2D(64, (3, 3), activation='relu'))
    cmodel.add(MaxPooling2D(pool_size=(2, 2), strides=None, padding='valid', data_format='channels_last'))

    cmodel.add(Conv2D(64, (3, 3), activation='relu'))
    cmodel.add(MaxPooling2D(pool_size=(2, 2), strides=None, padding='valid', data_format='channels_last'))

    cmodel.add(Conv2D(128, (3, 3), activation='relu'))
    cmodel.add(MaxPooling2D(pool_size=(2, 2), strides=None, padding='valid', data_format='channels_last'))

    cmodel.add(Flatten())
    cmodel.add(Dense(128, activation='relu'))
    cmodel.add(Dense(2, activation='softmax'))



    cmodel.compile(Adam(lr=.0001), loss='categorical_crossentropy', metrics=['accuracy'])
    cmodel.fit_generator(train_batches, steps_per_epoch=4, validation_data=valid_batches, validation_steps=15,
                         epochs=50, verbose=2)

    return cmodel
def augment_all():
    normal_image_path = "Data/train/normal/1.jpg"
    ab_image_path = "Data/train/abnormal/53.jpg"
    vnormal_image_path = "Data/valid/normal/96.jpg"
    vabnormal_image_path = "Data/valid/abnormal/76.jpg"
    testnormal_image_path = "Data/test/normal/92.jpg"
    testabnormal_image_path = "Data/test/abnormal/80.jpg"

    augment_images(normal_image_path, 50, "Data/train/normal/")
    augment_images(ab_image_path, 50, "Data/train/abnormal/")
    augment_images(vnormal_image_path, 50, "Data/valid/normal/")
    augment_images(vabnormal_image_path, 50, "Data/valid/abnormal/")
    augment_images(testnormal_image_path, 50, "Data/test/normal/")
    augment_images(testabnormal_image_path, 50, "Data/test/abnormal/")

if __name__ == '__main__':

    #augment_all()

    #model = train_model()
    #model2 = train_with_vgg()
    model3 = custom_model()
    model3.save_weights('weights.h5')
    model3.save('custom_model.h5')
    #model3 = load_model('my_model.h5')
    #test_model(model3)

