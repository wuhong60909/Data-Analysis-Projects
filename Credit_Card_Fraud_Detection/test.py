# %%
# Importing Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import matplotlib.gridspec as gridspec
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, \
classification_report, confusion_matrix, precision_recall_curve, \
average_precision_score, roc_curve, auc
from sklearn.model_selection import KFold, StratifiedKFold

# Importing dataset
dataset = pd.read_csv('creditcard.csv')
dataset_X = dataset.iloc[:, 0:30]
dataset_y = dataset.iloc[:, 30]

# Drop 'Time'
X = dataset_X.values
X = X[:, 1:] # Drop 'Time'
y = dataset_y.values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, shuffle = False)

# Feature Scaling: Only 'Amount'
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train[:, -1:] = sc.fit_transform(X_train[:, -1:])
X_test[:, -1:] = sc.transform(X_test[:, -1:])

# Fitting the classifier to the Training set: Logistic Regression
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(solver = 'liblinear', random_state = 0)
classifier.fit(X_train, y_train)

# %%
from curves import curves
a = curves(clf = classifier, X = X_test, y = y_test)
# %%
a.scores_plot()
a.ROC_plot()
a.PRF_plot()
a.PR_plot()
# %%
