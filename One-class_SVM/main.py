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

from sklearn.svm import OneClassSVM
from sklearn.metrics import auc
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import plot_precision_recall_curve
from sklearn.metrics import accuracy_score

#custome libaries
from data_preprocessing_IoT import IoT_data_common
#from Autoencoder_IoT_model import build_iot_AE


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
svm = OneClassSVM(kernel='rbf', gamma=1.0/train_data.shape[0], tol=0.001, nu=0.5, shrinking=True, cache_size=25)
svm = svm.fit(train_data)
ee_500 = time.time()
print('{:.3f} sec, Scikit Learn Normal'.format(ee_500-ss_500))

# scikit learn solution
ss_1000 = time.time()
svm_g_t = OneClassSVM(kernel='rbf', gamma=0.01, tol=0.01, nu=0.5, shrinking=True, cache_size=50)
svm_g_t = svm_g_t.fit(train_data)
ee_1000 = time.time()
print('{:.3f} sec, Scikit Learn Normal'.format(ee_1000-ss_1000))

#svm
scores = svm.decision_function(train_labels).flatten()
maxvalue = np.max(scores)
scores = maxvalue - scores
print("scores:\n", scores)

#svm_gama
scores_g_t = svm_g_t.decision_function(train_labels).flatten()
maxvalue = np.max(scores_g_t)
scores_g_t = maxvalue - scores_g_t
print("scores:\n", scores_g_t)

output = pd.DataFrame()

# perform reverse sort
sort_ix = np.argsort(scores)[::-1]
print("reverse sorting:\n", sort_ix)

output['labels'] =  test_label_original 
output['outlier_scores'] =  scores[sort_ix]

output.to_csv('/home/vibek/Anomanly_detection_packages/Unsupervised_learning/One-class_SVM/outlier_scores.csv', header=None, index=None)

output_f_t = pd.DataFrame()

# perform reverse sort
sort_ix_g_t = np.argsort(scores_g_t)[::-1]
print("reverse sorting:\n", sort_ix_g_t)

output_f_t['labels'] =  test_label_original 
output_f_t['outlier_scores'] =  scores_g_t [::-1]#[sort_ix_g_t]

output_f_t.to_csv('/home/vibek/Anomanly_detection_packages/Unsupervised_learning/One-class_SVM/outlier_scores_g_t.csv', header=None, index=None)

pr_curve_precision, pr_curve_recall, thresholds = precision_recall_curve(output['labels'], output['outlier_scores'])
pr_curve_precision_g_t, pr_curve_recall_g_t, thresholds_g_t = precision_recall_curve(output_f_t['labels'], output_f_t['outlier_scores'])

# sort so that recall is increasing
#ix = pr_curve_recall.argsort()
#pr_curve_recall = pr_curve_recall[ix]
#pr_curve_precision = pr_curve_precision[ix]
pr_curve_recall = pr_curve_recall[::-1]
pr_curve_precision = pr_curve_precision[::-1]

value = pd.DataFrame()
value ['recall'] = pr_curve_recall
value ['precision']= pr_curve_precision
value.to_csv('/home/vibek/Anomanly_detection_packages/Unsupervised_learning/One-class_SVM/pr_scores.csv', header=None, index=None)

value_g_t = pd.DataFrame()
value_g_t ['recall'] = pr_curve_recall_g_t[::-1]
value_g_t ['precision']= pr_curve_precision_g_t[::-1]
value_g_t.to_csv('/home/vibek/Anomanly_detection_packages/Unsupervised_learning/One-class_SVM/pr_scores_g_t.csv', header=None, index=None)

# pinning the initial precision to 1 is nonsense
# you should pin it to its first actual value
pr_curve_precision[0] = pr_curve_precision[1]

print("PR-AUC score %f" % auc(pr_curve_recall, pr_curve_precision))
print("PR-AUC scores_g_t %f" % auc(pr_curve_recall_g_t[::-1], pr_curve_precision_g_t[::-1]))


print("ROC-AUC score %f" % roc_auc_score(output['labels'], output['outlier_scores']))
print("ROC-AUC score %f" % roc_auc_score(output_f_t['labels'], output_f_t['outlier_scores']))


roc_curve_fpr, roc_curve_tpr, roc_curve_thresholds = roc_curve(output['labels'], output['outlier_scores'])
roc_curve_fpr_g_t, roc_curve_tpr_g_t, roc_curve_thresholds_g_t = roc_curve(output_f_t['labels'], output_f_t['outlier_scores'])

value_roc = pd.DataFrame()
value_roc ['fpr'] = roc_curve_fpr[::-1]
value_roc ['tpr']= roc_curve_tpr[::-1]
value_roc.to_csv('/home/vibek/Anomanly_detection_packages/Unsupervised_learning/One-class_SVM/roc_scores.csv', header=None, index=None)

value_roc_g_t = pd.DataFrame()
value_roc_g_t ['fpr'] = roc_curve_fpr[::-1]
value_roc_g_t ['tpr']= roc_curve_tpr[::-1]
value_roc_g_t.to_csv('/home/vibek/Anomanly_detection_packages/Unsupervised_learning/One-class_SVM/roc_scores_g_t.csv', header=None, index=None)

eta = pd.read_csv('/home/vibek/Anomanly_detection_packages/Unsupervised_learning/One-class_SVM/pr_scores.csv', header=None, index_col=False, skiprows=1)
eta_auc = auc(eta[0], eta[1])

eta_g_t = pd.read_csv('/home/vibek/Anomanly_detection_packages/Unsupervised_learning/One-class_SVM/pr_scores_g_t.csv', header=None, index_col=False, skiprows=1)
eta_auc_g_t = auc(eta_g_t[0], eta_g_t[1])

fig = plt.figure(figsize=(8,5))
ax = fig.add_axes([0.045, 0.1, 0.6, 0.8])
ax.plot(eta[0].values, eta[1].values, label='Test sample AUC=%f' % eta_auc, lw=2)
ax.plot(eta_g_t[0].values, eta_g_t[1].values, label='sklearn test sample AUC=%f' % eta_auc_g_t, lw=2)
#ax.plot(eta[0].values, eta[1].values, label='eta AUC=%f' % eta_auc_g_t_c, lw=2)
ax.set_xlabel('Recall', fontsize=10)
ax.set_ylabel('Precision', fontsize=10)
ax.set_ylim([0.0, 1.05])
ax.set_xlim([0.0, 1.0])
ax.set_title('SVM Precision-Recall', fontsize=10)
ax.legend(bbox_to_anchor=(0.6, 0.2), prop={'family': 'monospace'})
plt.show()


roc = pd.read_csv('/home/vibek/Anomanly_detection_packages/Unsupervised_learning/One-class_SVM/roc_scores.csv', header=None, index_col=False, skiprows=1)
roc_auc = auc(roc[0], roc[1])

roc_g_t = pd.read_csv('/home/vibek/Anomanly_detection_packages/Unsupervised_learning/One-class_SVM/roc_scores_g_t.csv', header=None, index_col=False, skiprows=1)
roc_auc_g_t = auc(roc_g_t[0], roc_g_t[1])

plt.figure()
lw = 2
plt.plot(eta[0].values, eta[1].values, color='red',
         lw=lw, label='ROC curve test sample (area = %0.2f)' % eta_auc)
plt.plot(eta_g_t[0].values, eta_g_t[1].values, color='black',
         lw=lw, label='ROC curve sklearn test sample (area = %0.2f)' % eta_auc_g_t)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.show()

'''
x = test['x1'] 
y = test['x2'] 
y_pred = svm.predict(test[['x1','x4','x5']]) 
colors = np.array(['#377eb8', '#ff7f00']) 
plt.scatter(x, y, alpha=0.7, c=colors[(y_pred + 1) // 2]) plt.xlabel('x1') 
plt.ylabel('x4')

'''