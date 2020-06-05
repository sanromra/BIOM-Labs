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

    if (ishape!=0):
        model.add(Conv2D(filters, (3, 3), padding='same',
        input_shape=ishape))
    else:
        model.add(Conv2D(filters, (3, 3), padding='same'))


    model.add(BN())
    model.add(GN(0.3))
    model.add(Activation('relu'))
    """
    model.add(Conv2D(filters, (3, 3), padding='same'))
    model.add(BN())
    model.add(GN(0.3))
    model.add(Activation('relu'))
    """
    model.add(MaxPooling2D(pool_size=(2, 2),name=lname))

    return model

def scheduler(epoch):
    if epoch < 100:
        return .0001
    elif epoch < 200:
        return 0.00001
    else:
        return 0.000001


def main():

    batch_size = 1024
    num_classes = 2
    epochs = 500


    train_x = np.load("train_X_FOURTH_iter5.npy", allow_pickle=True)
    train_y_orig = np.load("train_Y_FOURTH_iter5.npy", allow_pickle=True)
    #test_x = np.load("dev_x.npy")
    #test_y = np.load("dev_y.npy")
    print(train_x.shape)
    print(train_y_orig.shape)
    #print(test_x.shape)
    #print(test_y.shape)
    #for i in range(test_x.shape[0]):
        #print(test_x[i].shape)
        #test_x[i] = cv2.resize(test_x[i], dsize=(21,21), interpolation=cv2.INTER_CUBIC)
    train_x = train_x.astype("float32")
    #test_x = test_x.astype("float32")

    train_y = train_y_orig.astype("float32")
    #test_y = test_y.astype("float32")

    train_x /= 255
    #test_x /= 255

    num_classes = 2

    train_y = keras.utils.to_categorical(train_y, num_classes)
    #test_y = keras.utils.to_categorical(test_y, num_classes)

    datagen = ImageDataGenerator(
        width_shift_range=0.1,
        height_shift_range=0.1,
        rotation_range=10,
        #zoom_range=[1.0,1.1],
        #featurewise_std_normalization=True,
        horizontal_flip=True# , brightness_range=(0.95, 1.05)
    )
    datagen.fit(train_x)

    """
    model = Sequential()
    model = CBGN(model, 32, "conv_model1_1", train_x.shape[1:])
    model = CBGN(model, 64, "conv_model1_2")
    model = CBGN(model, 64, "conv_model1_3")
    model = CBGN(model, 128, "conv_model1_4")
    #model = CBGN(model, 128, "conv_model1_5")
    model.add(Flatten())
    model.add(Dropout(0.4, name="dropout_model1"))
    model.add(Dense(128))
    model.add(BN())
    model.add(GN(0.4))
    model.add(Activation("relu"))
    model.add(Dense(num_classes, activation="softmax"))

    model.summary()
    """
    """
    model = Sequential()
    model.add(Conv2D(8, (3, 3), padding='same', strides=1, input_shape=train_x.shape[1:]))
    model.add(Activation("relu"))#24
    #model.add(PReLU(alpha_initializer=Constant(value=0.25)))
    model.add(Conv2D(16, (3, 3), padding='valid', strides=2))
    model.add(Activation("relu"))#24
   # model.add(PReLU(alpha_initializer=Constant(value=0.25)))
    model.add(Conv2D(32, (3, 3), padding='valid', strides=1))
    model.add(Activation("relu"))#32
    #model.add(PReLU(alpha_initializer=Constant(value=0.25)))
    #model.add(Conv2D(32, (3, 3), padding='valid', strides=1))
    #model.add(Activation("relu"))#48
    #model.add(PReLU(alpha_initializer=Constant(value=0.25)))
    #model.add(Conv2D(32, (4, 3), padding='valid', strides=1))
    #model.add(Activation("relu"))#32
    #model.add(PReLU(alpha_initializer=Constant(value=0.25)))
    model.add(Conv2D(16, (3, 3), padding='valid', strides=1))
    model.add(Activation("relu"))#16
    #model.add(PReLU(alpha_initializer=Constant(value=0.25)))
    model.add(Conv2D(8, (3, 3), padding='valid', strides=1))
    model.add(Activation("relu"))
    #model.add(PReLU(alpha_initializer=Constant(value=0.25)))
    model.add(Flatten())
    model.add(GN(0.3))
    model.add(Dense(512, activation="relu"))
    model.add(Dense(num_classes, activation="softmax"))

    model.summary()
    """
    opt = SGD(lr=0.001, decay=1e-6, momentum=0.9)
    model = load_model("model_faces_20_iter6_dev.hdf5")
    
    model.compile(loss='categorical_crossentropy',optimizer=opt,metrics=['accuracy'])
    print(train_y.shape)   
    pesos = sklearn.utils.class_weight.compute_class_weight("balanced",np.asarray([0,1]), train_y_orig)
    print(str(pesos))

    set_lr = LRS(scheduler)
    save_model = keras.callbacks.ModelCheckpoint( "model_faces_20_iter6_dev_DA.hdf5", monitor='accuracy', verbose=1, save_best_only=True,  save_weights_only=False, mode='max')
    
    
    history=model.fit_generator(datagen.flow(train_x, train_y,batch_size),
                            steps_per_epoch=len(train_x) / batch_size,
                            epochs=epochs,# validation_data=(test_x, test_y),
                            callbacks=[save_model],
                            verbose=1)#, class_weight=pesos)
    """

    history=model.fit(train_x, train_y, batch_size=1024,
                            epochs=epochs,# validation_data=(test_x, test_y),
                            callbacks=[save_model],
                            verbose=1)#, class_weight=pesos)
     
    """
main()
