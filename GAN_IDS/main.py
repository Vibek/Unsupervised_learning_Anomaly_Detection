#numpy libraries
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import csv

from keras.optimizers import Adam, RMSprop

#custome libaries
from data_preprocessing_IoT import IoT_data_common
from Autoencoder_IoT_model import build_iot_AE
import model

params = {'dataset': 'IoT-23'}
result_path="/home/vibek/Anomanly_detection_packages/Unsupervised_learning/"

###calling IoT-23 dataset####
print("Loading dataset IoT-23.....\n")
train_data, train_labels, test_data, test_labels = IoT_data_common(params)
print("train shape: ", train_data.shape)
print("test shape: ", test_data.shape)
print("train_label shape: ", train_labels.shape)
print("test_label shape: ", test_labels.shape)

update_test = train_labels[0]
print("test shape\n:", update_test.shape)

'''
test_label_original = np.argmax(test_labels, axis=1)
train_label_original = np.argmax(test_data, axis=1)

BATCH_SIZE = 128

#train GAN model
#Model_d, Model_g = model.train(BATCH_SIZE, train_data)


## generate random tarffic 
generated_data = model.generate(55321)
print("generated_traffic:\n", generated_data)

myFile = open('/home/vibek/Anomanly_detection_packages/Unsupervised_learning/generated_data.csv', 'w')
with myFile:
   writer = csv.writer(myFile)
   writer.writerows(generated_data)

## compute anomaly score - sample from test set
loss_list = []
#y_list = []
#for i, (x,y) in enumerate(test_dataloader):
#    print(i, y)
    
#z = np.random.uniform(0, 1, size=(BATCH_SIZE, 11))
#z = np.zeros(z)
generated_data = model.generate(55321)
generate_optimizer = Adam(lr=0.0002, beta_1=0.5)

loss = None
for j in range(100): # set your interation range
	#gen_fake,_ = model.generator_model(z)
	loss = model.Anomaly_score(train_labels, generated_data)
	loss.backward()
	generate_optimizer.step()

loss_list.append(loss) # Store the loss from the final iteration
#y_list.append(y) # Store the corresponding anomaly label
print('~~~~~~~~loss={},  y={} ~~~~~~~~~~'.format(loss, y))
#break
print("anomaly score : " + str(loss))
'''