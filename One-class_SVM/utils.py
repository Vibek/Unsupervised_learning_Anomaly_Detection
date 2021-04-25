import numpy as np
import pandas as pd
from sklearn.metrics import auc
from sklearn.metrics import precision_recall_curve

# group by the first column
df['binary_truth'] = 0
grouped = df.groupby(0)
outliers = grouped.get_group('Outlier')
df['binary_truth'].iloc[outliers.index] = 1

pr_curve_precision, pr_curve_recall, thresholds = precision_recall_curve(df['binary_truth'].values, df[1].values)

# sort so that recall is increasing
#ix = pr_curve_recall.argsort()
#pr_curve_recall = pr_curve_recall[ix]
#pr_curve_precision = pr_curve_precision[ix]
pr_curve_recall = pr_curve_recall[::-1]
pr_curve_precision = pr_curve_precision[::-1]

# pinning the initial precision to 1 is nonsense
# you should pin it to its first actual value
pr_curve_precision[0] = pr_curve_precision[1]

print("PR-AUC score %f" % auc(pr_curve_recall, pr_curve_precision))

for i in range(0, pr_curve_recall.shape[0]):
    print("%f, %f" % (pr_curve_recall[i], pr_curve_precision[i]))



print("ROC-AUC score %f" % roc_auc_score(df['binary_truth'].values, df[1].values))

roc_curve_fpr, roc_curve_tpr, roc_curve_thresholds = roc_curve(df['binary_truth'].values, df[1].values)
for i in range(0, roc_curve_fpr.shape[0]):
    print("%f, %f" % (roc_curve_fpr[i], roc_curve_tpr[i]))