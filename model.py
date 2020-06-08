import os
import keras
from keras.models import Sequential
from keras.layers import Activation, Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from keras.optimizers import Adam
from keras.callbacks import TensorBoard
from keras.callbacks import LearningRateScheduler as LRS
from keras.preprocessing.image import ImageDataGenerator
from keras.layers.normalization import BatchNormalization as BN
from keras.layers import GaussianNoise as GN
from keras.initializers import Constant
from keras.layers import PReLU
from keras.optimizers import SGD
from keras.models import load_model
import keras.backend as K

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import sklearn
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import accuracy_score
from keras.utils import np_utils
import itertools
import cv2

os.environ['CUDA_DEVICE_ORDER']="PCI_BUS_ID";
os.environ['CUDA_VISIBLE_DEVICES']="0";


def main():

    batch_size = 1024
    num_classes = 2
    epochs = 500


    #Load created npy arrays with vectorized images
    train_x = np.load("train_X_FOURTH_iter5.npy", allow_pickle=True)

    #Load labels
    train_y_orig = np.load("train_Y_FOURTH_iter5.npy", allow_pickle=True)

    train_x = train_x.astype("float32")
    train_y = train_y_orig.astype("float32")

    #Normalize data
    train_x /= 255

    #One-hot vector of labels
    num_classes = 2
    train_y = keras.utils.to_categorical(train_y, num_classes)


    #Data augmentation on the fly
    datagen = ImageDataGenerator(
        width_shift_range=0.1, #horizontal shift
        height_shift_range=0.1, #Vertical shift
        rotation_range=10, #Image rotation [-10, +10] degrees
        horizontal_flip=True #Flip the image
    )

    #Train the datagenerator over the train set
    datagen.fit(train_x)


    """
    
    ###### CONSTRUCTION OF MODEL ######

    """
    model = Sequential()
    model.add(Conv2D(8, (3, 3), padding='same', strides=1, input_shape=train_x.shape[1:]))
    model.add(Activation("relu"))

    model.add(Conv2D(16, (3, 3), padding='valid', strides=2))
    model.add(Activation("relu"))

    model.add(Conv2D(32, (3, 3), padding='valid', strides=1))
    model.add(Activation("relu"))

    model.add(Conv2D(16, (3, 3), padding='valid', strides=1))
    model.add(Activation("relu"))

    model.add(Conv2D(8, (3, 3), padding='valid', strides=1))
    model.add(Activation("relu"))

    model.add(Flatten())
    model.add(GN(0.3))
    model.add(Dense(512, activation="relu"))
    model.add(Dense(num_classes, activation="softmax"))

    model.summary()
    
    opt = SGD(lr=0.001, decay=1e-6, momentum=0.9)

    #Only if fine tunning a previously trained model. If so, reduce learning rate above.
    #model = load_model("model_faces_20_iter6_dev.hdf5")
    
    model.compile(loss='categorical_crossentropy',optimizer=opt,metrics=['accuracy'])

    save_model = keras.callbacks.ModelCheckpoint( "model_faces_20_iter6_dev_DA.hdf5", monitor='accuracy', verbose=1, save_best_only=True,  save_weights_only=False, mode='max')
    
    
    history=model.fit_generator(datagen.flow(train_x, train_y,batch_size),
                            steps_per_epoch=len(train_x) / batch_size,
                            epochs=epochs,
                            callbacks=[save_model],
                            verbose=1)
   
main()
