import numpy as np
import pandas as pd
import random as rn
import time
import jgraph as ig
import random as rn
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style(style="whitegrid")
sns.set_color_codes()


#custome libaries
from data_preprocessing_IoT import IoT_data_common
#from Autoencoder_IoT_model import build_iot_AE
from encoder import EncoderForest

import utils

params = {'dataset': 'IoT-23'}

###calling IoT-23 dataset####
print("Loading dataset IoT-23.....\n")
train_data, train_labels, test_data, test_labels = IoT_data_common(params)
print("train shape: ", train_data.shape)
print("test shape: ", test_data.shape)
print("train_label shape: ", train_labels.shape)
print("test_label shape: ", test_labels.shape)

test_label_original = np.argmax(test_labels, axis=1)

# scikit learn solution
ss_500 = time.time()
encoder = EncoderForest(500)
encoder.fit(train_data, max_depth=10)
print("end fit")
encoded = encoder.encode(train_data)
print("end encode")
ee_500 = time.time()
print('{:.3f} sec, Scikit Learn Normal'.format(ee_500-ss_500))

# scikit learn solution
ss_1000 = time.time()
encoder_1k = EncoderForest(1000)
encoder_1k.fit(train_data, max_depth=10)
print("end fit")
encoded_1k = encoder_1k.encode(train_data)
print("end encode")
ee_1000 = time.time()
print('{:.3f} sec, Scikit Learn Normal'.format(ee_1000-ss_1000))

# scikit learn solution
ss_2000 = time.time()
encoder_2k = EncoderForest(2000)
encoder_2k.fit(train_data, max_depth=10)
print("end fit")
encoded_2k = encoder_2k.encode(train_data)
print("end encode")
ee_2000 = time.time()
print('{:.3f} sec, Scikit Learn Normal'.format(ee_2000-ss_2000))

img_prime_1k = encoder_1k.decode(encoded_1k[100000])#.reshape(10, 10)
print("end decode",img_prime_1k)

img_prime_2k = encoder_2k.decode(encoded_2k[10000])#.reshape(10, 10)
print("end decode",img_prime_2k)

img_prime = encoder.decode(encoded[100078])#.reshape(10, 10)
print("end decode",img_prime)

ss1=np.argsort(img_prime)
print("\n decoded result \n:", ss1)

ss2=np.argsort(img_prime_1k)
print("\n decoded result \n:", ss2)

ss3=np.argsort(img_prime_2k)
print("\n decoded result \n:", ss3)

f = plt.figure(figsize=(20,10))
plt.subplot(1,3,1)
sns.distplot(img_prime, kde=True, color="r")
plt.title('EncoderForest with 500 trees and depth 20')
plt.xlabel('Reconstruction Error')

plt.subplot(1,3,2)
sns.distplot(img_prime_1k, kde=True, color="k")
plt.title('EncoderForest with 1k trees and depth 20')
plt.xlabel('Reconstruction Error')

plt.subplot(1,3,3)
sns.distplot(img_prime_2k, kde=True, color="b")
plt.title('EncoderForest with 2k trees and depth 20')
plt.xlabel('Reconstruction Error')
plt.show()

#sns.countplot(encoder.decode(encoded[100000]))

'''
df = pd.DataFrame(data=train_data)
f = plt.figure(figsize=(20,10))
plt.subplot(1,1,1)
sns.pairplot(df, diag_kind="kde")
plt.title('Distribution plot')
plt.show()
'''

Sorted=True
fig = plt.figure(figsize=(20,10))
ax1 = plt.subplot(122, projection='polar')
rn, thetan = utils.getVals(encoded,np.array([0.,0.]),sorted=Sorted)
for j in range(len(rn)):
    ax1.plot([thetan[j],thetan[j]], [1,rn[j]], color='b',alpha=1,lw=1)

ra, thetaa = utils.getVals(encoded,np.array([3.3,3.3]),sorted=Sorted)
for j in range(len(ra)):
    ax1.plot([thetaa[j],thetaa[j]], [1,ra[j]], color='r',alpha=0.9,lw=1.3)
    
ax1.set_title("Normal Isolation Forest\nNormal: Mean={0:.3f}, Var={1:.3f}\nAnomaly: Mean={2:.3f}, Var={3:.3f}".format(np.mean(rn),np.var(rn),np.mean(ra),np.var(ra)))

ax1.set_xticklabels([])
ax1.set_xlabel("Anomaly")
ax1.set_ylim(0,encoded.limit)

ax1.axes.get_xaxis().set_visible(False)
ax1.axes.get_yaxis().set_visible(False)
plt.show()
