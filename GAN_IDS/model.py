from __future__ import print_function

#numpy libraries
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import tensorflow as tf
import math
import time
from datetime import datetime
from timeit import default_timer as timer

#Keras libraries
from keras.models import Sequential, Model
from keras.layers import Input, Reshape, Dense, Dropout, MaxPooling2D, Conv2D, LSTM, Flatten, UpSampling2D, Conv2D
from keras.layers import Conv2DTranspose, LeakyReLU
from keras.layers.core import Activation
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam, RMSprop
from keras import backend as K
from keras import initializers
from keras.utils. generic_utils import Progbar
from keras.utils import plot_model
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, CSVLogger, TensorBoard, LearningRateScheduler, ModelCheckpoint, Callback

#TensorFlow libraries
from tensorflow.keras import callbacks
from tensorflow.keras import backend as K
import tensorflow as tf

#custome libaries
from data_preprocessing_IoT import IoT_data_common
from Autoencoder_IoT_model import build_iot_AE
import utils


params = {'dataset': 'IoT-23'}
result_path="/home/vibek/Anomanly_detection_packages/Unsupervised_learning/"


#calling fearture reduction algorithm
print("Loading AutoEncoder model....\n")
autoencoder_1, encoder_1, autoencoder_2, encoder_2, autoencoder_3, encoder_3, sSAE, SAE_encoder = build_iot_AE()
print("value of SAE_encoder output:\n", SAE_encoder.output)
print("value of SAE_encoder input:\n", SAE_encoder.input)


### generator model define
def generator_model():
    input_traffic = Input(shape=(11, ))
    fc1 = Dense(units=50, activation='relu')(input_traffic)
    fc1 = BatchNormalization()(fc1)
    fc1 = LeakyReLU(0.2)(fc1)
    #fc2 = Reshape((1, 32))(fc1)
    up1 = Dense(units=32, activation='relu')(fc1)
    conv1 = BatchNormalization()(up1)
    #conv1 = Reshape((1, 16))(conv1)
    up2 = Dense(units=26, activation='relu')(conv1)
    conv2 = BatchNormalization()(up2)
    #conv2 = Reshape((1, 16))(conv2)
    conv3 = Dense(units=11, activation='relu')(conv2)
        
    model = Model(input_traffic, conv3)

    plot_model(model,to_file='/home/vibek/Anomanly_detection_packages/Unsupervised_learning/model_generator_GAN.png',show_shapes=True)
    #model.summary()
    return model

### discriminator model define
def discriminator_model():
    #inputs = Input((28, 28, 1))
    fc1 = Dense(units=64, activation='relu')(SAE_encoder.output)
    fc1 = Dropout(0.3)(fc1)
    fc1 = BatchNormalization()(fc1)
    fc1 = LeakyReLU(0.2)(fc1)
    fc2 = Reshape((1, 64))(fc1)
    fc2 = LSTM(units=50, activation='tanh', return_sequences=True)(fc2)
    fc2 = Dropout(0.3)(fc2)
    fc3 = Reshape((1, 50))(fc2)
    fc4 = LSTM(units=2)(fc3)
    outputs = Activation('sigmoid')(fc4)
    
    model = Model(SAE_encoder.input, outputs)

    plot_model(model,to_file='/home/vibek/Anomanly_detection_packages/Unsupervised_learning/model_discriminator_GAN.png',show_shapes=True)
    #model.summary()

    opt = Adam(lr=0.0002, beta_1=0.5)
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
    return model

### d_on_g model for training generator
def gan_model(g, d):
    d.trainable = False
    ganInput = Input(shape=(11,))
    x = g(ganInput)
    print("Vaue of x:\n", x)

    ganOutput = d(x)
    print("GAN output:\n", ganOutput)
    gan = Model(ganInput, ganOutput)

    plot_model(gan,to_file='/home/vibek/Anomanly_detection_packages/Unsupervised_learning/model_GAN.png',show_shapes=True)
    gan.summary()

    #gan.compile(loss='binary_crossentropy', optimizer='adam')
    return gan

### train generator and discriminator
def train(BATCH_SIZE, X_train):
    
    ### model define
    d = discriminator_model()
    print("#### discriminator ######")
    d.summary()
    g = generator_model()
    print("#### generator ######")
    g.summary()
    d_on_g = gan_model(g, d)
    d_optim = RMSprop(lr=0.0004)
    g_optim = RMSprop(lr=0.0002)
    g.compile(loss='mse', optimizer=g_optim)
    d_on_g.compile(loss='mse', optimizer=g_optim)
    d.trainable = True
    d.compile(loss='mse', optimizer=d_optim)
    

    for epoch in tqdm(range(100)):
        print("Epoch", str(epoch) + "/" + str(100))
        n_iter = int(X_train.shape[0]/BATCH_SIZE)
        progress_bar = Progbar(target=n_iter)
        
        for index in range(n_iter):

            # create random noise -> U(0,1) 11 latent vectors
            noise = np.random.uniform(0, 1, size=(BATCH_SIZE, 11))

            # load real data & generate fake data
            traffic_batch = X_train[index*BATCH_SIZE:(index+1)*BATCH_SIZE]
            generated_traffic = g.predict(noise, verbose=0)
            
            # visualize training results
            #print("\n generate results:\n", generated_traffic)

            # attach label for training discriminator
            X = np.concatenate((traffic_batch, generated_traffic))
            y = np.array([1] * BATCH_SIZE + [0] * BATCH_SIZE)
            
            # training discriminator
            d_loss = d.train_on_batch(X, y)

            # training generator
            d.trainable = False
            g_loss = d_on_g.train_on_batch(noise, np.array([1] * BATCH_SIZE))
            d.trainable = True

            progress_bar.update(index, values=[('g',g_loss), ('d',d_loss)])
        print ('')

        # save weights for each epoch
        g.save_weights('/home/vibek/Anomanly_detection_packages/Unsupervised_learning/model/generator.h5', True)
        d.save_weights('/home/vibek/Anomanly_detection_packages/Unsupervised_learning/model/discriminator.h5', True)
    return d, g

### generate 
def generate(BATCH_SIZE):
    g = generator_model()
    g.load_weights('/home/vibek/Anomanly_detection_packages/Unsupervised_learning/model/generator.h5')
    noise = np.random.uniform(0, 1, (BATCH_SIZE, 11))
    generated_traffic = g.predict(noise)
    return generated_traffic


### anomaly custome loss function 
def sum_of_residual(y_true, y_pred):
    return tf.reduce_sum(abs(y_true - y_pred))


# Lambda = 0.1 is set in this work
# x is new data, G_z is closely regenerated data

def Anomaly_score(x, G_z, Lambda=0.1):
    residual_loss = sum_of_residual(x, G_z) # Residual Loss

    d = discriminator_model()
    d.load_weights('/home/vibek/Anomanly_detection_packages/Unsupervised_learning/model/discriminator.h5') 
    #d.trainable = False
    
    # x_feature is a rich intermediate feature representation for real data x
    output_x= d(x) 
    x_feature = d.predict(x)
    # G_z_feature is a rich intermediate feature representation for fake data G(z)
    output_G_z = d(G_z) 
    G_z_feature = d.predict(G_z)
    discrimination_loss = sum_of_residual(x_feature, G_z_feature) # Discrimination loss
    
    total_loss = (1-Lambda)*residual_loss + Lambda*discrimination_loss
    return total_loss









'''
### discriminator intermediate layer feautre extraction
def feature_extractor():
   
    d = discriminator_model()
    d.load_weights('/home/vibek/Anomanly_detection_packages/Unsupervised_learning/model/discriminator.h5') 
    intermidiate_model = Model(d.layers[0].input, d.layers[-7].output)
    #intermidiate_model.summary()
    intermidiate_model.compile(loss='binary_crossentropy', optimizer='rmsprop')
    return intermidiate_model

### anomaly detection model define
def anomaly_detector():
    #if g is None:
    g = generator_model()
    g.load_weights('/home/vibek/Anomanly_detection_packages/Unsupervised_learning/model/generator.h5')
    g.trainable = False
    #intermidiate_model = feature_extractor()
    #print("intermidiate_model feature:\n", intermidiate_model)
    #intermidiate_model.trainable = False
    #g = Model(inputs=g.layers[1].input, outputs=g.layers[-1].output)
    #g.trainable = False
    # Input layer cann't be trained. Add new layer as same size & same distribution
    #aInput = Input(shape=(11,))
    gInput = Dense(units=11, activation='sigmoid')(SAE_encoder.output)
    #gInput = Activation('sigmoid')(gInput)
    
    # G & D feature
    G_out = g(gInput)
    #D_out= intermidiate_model(G_out)    
    model = Model(SAE_encoder.input, G_out)#[G_out, D_out]
    model.compile(loss=sum_of_residual, loss_weights=0.9, optimizer='adam')#[0.90, 0.10]
    
    # batchnorm learning phase fixed (test) : make non trainable
    #K.set_learning_phase(0)
    
    return model

### anomaly detection score
def compute_anomaly_score(model, x):
    z = np.random.uniform(0, 1, size=(55321,))
    print("values of Z:\n", z.shape)
    #g = generator_model()
    #gen_fake,_ = g(z)
    #print("Gner fake traffic\n:", gen_fake)
    
    #d = discriminator_model()
    intermidiate_model = feature_extractor()
    d_x = intermidiate_model.predict(x)
    print("values of d_x:\n", d_x.shape)
	
    # learning for changing latent
    loss = model.fit(x, z, batch_size=64, epochs=10, verbose=0) #[x, d_x]
    similar_data, _ = model.predict(z)
    print("similar_data:\n", similar_data)
    
    loss = loss.history['loss'][-1]
    
    return loss, similar_data
    '''