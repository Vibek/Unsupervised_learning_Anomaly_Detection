from __future__ import print_function

from datetime import datetime
import pandas as pd
import numpy as np
import io
import itertools
import pickle
import shutil
import time
import h5py

from sklearn import metrics
import matplotlib.pyplot as plt

from keras.models import Model
from keras.utils import plot_model
from keras.optimizers import Adam
from keras.losses import binary_crossentropy
from keras.layers import Dense, Reshape, Dropout, Input
from keras.layers.normalization import BatchNormalization
from keras.preprocessing.sequence import TimeseriesGenerator
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.datasets import imdb
from tensorflow.keras import callbacks
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, CSVLogger, TensorBoard, LearningRateScheduler, ModelCheckpoint, Callback
from tensorflow import keras
from tensorflow.keras import backend as k

#custome libaries
from data_preprocessing_IoT import IoT_data_common
from Autoencoder_IoT_model import build_iot_AE

params = {'dataset': 'IoT-23'}

###IoT-23 dataset####
print("Loading dataset IoT-23.....\n")
train_data, train_labels, test_data, test_labels = IoT_data_common(params)

#calling fearture reduction algorithm
print("Loading AutoEncoder model....\n")
autoencoder_1, encoder_1, autoencoder_2, encoder_2, autoencoder_3, encoder_3, sSAE, SAE_encoder = build_iot_AE()
print("value of SAE_encoder output:\n", SAE_encoder.output)
print("value of SAE_encoder input:\n", SAE_encoder.input)

output_dim = 2
batch_size = 128

def build_model_black_box():
# 1. define the network
	mlp0 = Dense(units=32, activation='relu')(SAE_encoder.output)
	mlp0_drop = Dropout(0.3)(mlp0)

	mlp1 = Dense(units=16, activation='relu')(mlp0_drop)
	mlp_drop1 = Dropout(0.3)(mlp1)

	mlp2 = Dense(units=10, activation='relu')(mlp_drop1)
	mlp_drop2 = Dropout(0.3)(mlp2)

	mlp3 = Dense(units=6, activation='relu')(mlp_drop2)
	mlp_drop3 = Dropout(0.3)(mlp3)

	mlp4 = Dense(units=output_dim, activation='sigmoid')(mlp_drop3)

	model = Model(SAE_encoder.input, mlp4)
	model.summary()
	plot_model(model,to_file='/home/vibek/Anomanly_detection_packages/Unsupervised_learning/model_black_box.png',show_shapes=True)

	return model

model = build_model_black_box()
opt = Adam(lr=0.0002, beta_1=0.5)

# try using different optimizers and different optimizer configs
start = time.time()
model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
print ("Compilation Time:", time.time() - start)
#plot_losses = PlotLossesCallback()

#save model and the values
early_stopping_monitor = EarlyStopping(monitor='val_loss', patience=50)
callbacks = [early_stopping_monitor]

global_start_time = time.time()
print("Start Training...")
model.fit(train_data, test_data, batch_size=batch_size, epochs=100, validation_data=(train_labels, test_labels), callbacks=callbacks)
model.save("/home/vibek/Anomanly_detection_packages/Unsupervised_learning/model_black_box.h5")
print("Done Training...")