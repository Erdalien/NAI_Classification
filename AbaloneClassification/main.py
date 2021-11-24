import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import sklearn
from sklearn import  linear_model, dummy, metrics
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import  train_test_split
from sklearn.metrics import *
from sklearn.cluster import KMeans

#naming columns for dataset
column_names = ["sex", "length", "diameter", "height", "whole weight",
               "shucked weight", "viscera weight", "shell weight", "rings"]

#dataset
df = pd.read_csv("abalone.data", names=column_names)

x = df[['length', 'diameter', 'height', 'whole weight', 'shucked weight',
        'viscera weight', 'shell weight', 'rings']] #predictors

y = df['sex'] #what we are looking for

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

logisticRegr = LogisticRegression(solver='liblinear', fit_intercept=True)
logisticRegr.fit(x_train, y_train)
logis_pred = logisticRegr.predict(x_test)
logis_pred_prob = logisticRegr.predict_proba(x_test)

dummy = DummyClassifier(strategy= 'most_frequent')
dumb = dummy.fit(x_train, y_train)
dumb_pred = dumb.predict(x_test)
dumb_pred_prob = dumb.predict_proba(x_test)

cm_dummy = confusion_matrix(y_test, dumb_pred)
cm_dummy_df = pd.DataFrame(cm_dummy)
'''
Heatmap plot for dummy
fix, ax = plt.subplots(figsize = (7,7))
sns.heatmap(cm_dummy_df.T, annot=True, annot_kws={"size": 15}, cmap="Oranges", vmin=0, vmax=800,
            fmt='.0f', linewidths=1, linecolor="white", cbar=False, xticklabels=["Male", "Infant", 'Female'],
            yticklabels=["Male", "Infant", 'Female'])
plt.ylabel("Predicted", fontsize=15)
plt.xlabel("Actual", fontsize=15)
ax.set_xticklabels(["Male", "Infant", 'Female'], fontsize=13)
ax.set_yticklabels(["Male", "Infant", 'Female'], fontsize=13)

plt.show()
'''

cm_logis = confusion_matrix(y_test, logis_pred)
cm_logis_df = pd.DataFrame(cm_logis)

fix, ax = plt.subplots(figsize = (7,7))
sns.heatmap(cm_logis_df.T, annot=True, annot_kws={"size": 15}, cmap="Oranges", vmin=0, vmax=800,
            fmt='.0f', linewidths=1, linecolor="white", cbar=False, xticklabels=["Male", "Infant", 'Female'],
            yticklabels=["Male", "Infant", 'Female'])
plt.ylabel("Predicted", fontsize=15)
plt.xlabel("Actual", fontsize=15)
ax.set_xticklabels(["Male", "Infant", 'Female'], fontsize=13)
ax.set_yticklabels(["Male", "Infant", 'Female'], fontsize=13)

plt.show()