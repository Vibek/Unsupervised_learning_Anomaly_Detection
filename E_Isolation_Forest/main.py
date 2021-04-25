import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random as rn
import eif as iso
import eif_old as iso_old
import ex_if
import time
import jgraph as ig

import scipy.ndimage
from scipy.interpolate import griddata
import numpy.ma as ma
from numpy.random import uniform, seed
from sklearn.ensemble import IsolationForest
from sklearn.manifold import TSNE
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import multivariate_normal
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import recall_score
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer
from sklearn.metrics import accuracy_score

# Set some parameters to get good visuals - style to ggplot and size to 15,10
plt.style.use('ggplot')
import matplotlib.style as style
import seaborn as sb
sb.set_style(style="whitegrid")
sb.set_color_codes()

#custome libaries
from data_preprocessing_IoT import IoT_data_common
#from Autoencoder_IoT_model import build_iot_AE
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
ss = time.time()

Ex_IF_E  = iso.iForest(train_data, ntrees=2500, sample_size=2048, ExtensionLevel=1)

Ex_IF_N  = iso.iForest(train_data, ntrees=2000, sample_size=2048, ExtensionLevel=0)

ee = time.time()

print('{:.3f} sec, Scikit Learn Extended'.format(ee-ss))

ss = time.time()

clfIF = IsolationForest(max_samples=0.25, random_state=11, contamination=0.06, n_estimators=1000, n_jobs=-1)
clfIF.fit(train_data, test_data)

ee = time.time()

print('{:.3f} sec, Scikit Learn Normal'.format(ee-ss))

Score_data_N = clfIF.score_samples(train_data)*-1
print("\n Normal IsolationForest score_samples \n:", Score_data_N)

Score_data_E = Ex_IF_E.compute_paths(X_in=train_data)
print("\n Extended IsolationForest score_samples \n:", Score_data_E)


scores_pred_N = clfIF.decision_function(train_labels)

print("\nPrediction score:\n", scores_pred_N)

y_pred_train = clfIF.predict(train_labels)

np.unique(y_pred_train)

print(y_pred_train)

pred = np.array(['Anomaly' if i==1 else 'Normal' for i in y_pred_train])

dict_ = {'anomaly_score':scores_pred_N, 'y_pred':y_pred_train, 'result':pred}
scores = pd.DataFrame(dict_)
print(scores.sample(1000))


ss0=np.argsort(Score_data_N)
print("\n result \n:", ss0)

ss1=np.argsort(Score_data_E)
print("\n result \n:", ss1)


f = plt.figure(figsize=(20,10))
plt.subplot(1,2,1)
sb.distplot(Score_data_N, kde=True, color="r")
plt.title('Normal IsolationForest')
plt.xlabel('Anomaly Scores')
#plt.show()

plt.subplot(1,2,2)
sb.distplot(Score_data_E, kde=True, color="r")
plt.title('Modified IsolationForest')
plt.xlabel('Anomaly Scores')
plt.show()


'''
plt.scatter(train_data[0],train_data[1],s=15,c='b',edgecolor='b')
plt.scatter(train_data[ss1[-10000:]],train_data[ss1[-10000:]],s=55,c='g')
plt.scatter(train_data[ss1[:10000]],train_data[ss1[:10000]],s=55,c='r')
plt.title('Extended IsolationForest')


plt.subplot(1,1,1)
sb.kdeplot(Score_data, lw=3, label='Ex_if Scores', clip=(0.35, 0.8))
plt.legend(loc=0)
plt.xlabel('Scores')
plt.show()
'''

Sorted=True
fig = plt.figure(figsize=(20,10))
ax1 = plt.subplot(121, projection='polar')
rn, thetan = utils.getVals(Ex_IF_E,np.array([0.,0.]),sorted=Sorted)
for j in range(len(rn)):
    ax1.plot([thetan[j],thetan[j]], [1,rn[j]], color='b',alpha=1,lw=1)

ra, thetaa = utils.getVals(Ex_IF_E,np.array([3.3,3.3]),sorted=Sorted)
for j in range(len(ra)):
    ax1.plot([thetaa[j],thetaa[j]], [1,ra[j]], color='r',alpha=0.9,lw=1.3)
    
ax1.set_title("Modified Isolation Forest\nNormal: Mean={0:.3f}, Var={1:.3f}\nAnomaly: Mean={2:.3f}, Var={3:.3f}".format(np.mean(rn),np.var(rn),np.mean(ra),np.var(ra)))

ax1.set_xticklabels([])
ax1.set_xlabel("Anomaly")
ax1.set_ylim(0,Ex_IF_E.limit)

ax1.axes.get_xaxis().set_visible(False)
ax1.axes.get_yaxis().set_visible(False)
#plt.show()

Sorted=True
#fig = plt.figure(figsize=(20,10))
ax1 = plt.subplot(122, projection='polar')
rn, thetan = utils.getVals(Ex_IF_N,np.array([0.,0.]),sorted=Sorted)
for j in range(len(rn)):
    ax1.plot([thetan[j],thetan[j]], [1,rn[j]], color='b',alpha=1,lw=1)

ra, thetaa = utils.getVals(Ex_IF_N,np.array([3.3,3.3]),sorted=Sorted)
for j in range(len(ra)):
    ax1.plot([thetaa[j],thetaa[j]], [1,ra[j]], color='r',alpha=0.9,lw=1.3)
    
ax1.set_title("Normal Isolation Forest\nNormal: Mean={0:.3f}, Var={1:.3f}\nAnomaly: Mean={2:.3f}, Var={3:.3f}".format(np.mean(rn),np.var(rn),np.mean(ra),np.var(ra)))

ax1.set_xticklabels([])
ax1.set_xlabel("Anomaly")
ax1.set_ylim(0,Ex_IF_N.limit)

ax1.axes.get_xaxis().set_visible(False)
ax1.axes.get_yaxis().set_visible(False)
plt.show()

 # To encode string  labels into numbers
score = accuracy_score(test_label_original, y_pred_train)
print("Accuracy_score:", score)

print('Classification Report:\n')
print(classification_report(test_label_original, y_pred_train, target_names=['unknown','normal', 'anomaly']))
print ("AUC: ", "{:.1%}".format(roc_auc_score(test_label_original, y_pred_train)))

#scoring = {'AUC': 'roc_curve', 'Recall': make_scorer(recall_score, pos_label=-1)}
'''
n_v = [0]
n_e = []
jt=0
T=Ex_IF_N.Trees[jt]
P=iso.PathFactor(train_data[ss1[-1]],T)
Gn=branch2num(P.path_list)
lb=gg.get_shortest_paths('0_'+str(Gn[0]), '0_'+str(Gn[-1]))[0]
le=gg.get_eids([(lb[i],lb[i+1]) for i in range(len(lb)-1)])
vstyle2 = copy.deepcopy(vstyle)
for j in le: 
    vstyle2["edge_color"][j]= 'blue'
    vstyle2["edge_width"][j] = 1.9
for v in lb:
    vstyle2["vertex_color"][v]='blue'
ig.plot(gg,**vstyle2)
'''