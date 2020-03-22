# -*- coding: utf-8 -*-
"""
Created on Sun Mar 22 22:34:17 2020

@author: Hong
"""

# %% 
# Importing Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import precision_recall_curve, classification_report, confusion_matrix, average_precision_score, roc_curve, auc

# %% 
# Importing dataset
dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values

# %% 
# Data Preprocessing
# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])
labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])
onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X).toarray()
X = X[:, 1:] # Avoiding dummy variable trap! 

# %% 
# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# %% 
# Fitting the classifier to the Training set: Logistic Regression
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(solver = 'liblinear', random_state = 0)
classifier.fit(X_train, y_train)
classifier.intercept_
classifier.coef_

# %%
# Training set
train_score = pd.DataFrame({'score': classifier.predict_proba(X_train)[:, 1], 
                            'True_class': y_train})

# %%
train_score.describe()

# %%
plt.figure(figsize = (12, 5))
plt.scatter(train_score.index[train_score['True_class'] == 0], 
            train_score[train_score['True_class'] == 0]['score'], 
            s = 5, label = 'Normal')
plt.scatter(train_score.index[train_score['True_class'] == 1], 
            train_score[train_score['True_class'] == 1]['score'], 
            s = 5, label = 'Exit the Bank')
plt.xlabel('Index')
plt.ylabel('Score')
plt.legend()
plt.show()

# %% 
# Plotting the precision recall curve.
precision, recall, threshold = precision_recall_curve(train_score.True_class, train_score.score)
f1_score = 2 * precision * recall / (precision + recall)
average_precision = average_precision_score(train_score.True_class, train_score.score)

# %% 
# Choosing the threshold to maximize the F1 score
max_f1 = f1_score[f1_score == max(f1_score)]
best_threshold = threshold[f1_score[1: ] == max_f1]

# %% 
# Precision, Recall curve 
plt.figure(figsize = (12, 6))
plt.plot(threshold, precision[1: ], label = "Precision", linewidth = 3)
plt.plot(threshold, recall[1: ], label ="Recall", linewidth = 3)
plt.axvline(best_threshold, color = 'black', ls = '--', label = 'Threshold = %0.3f' % (best_threshold))
plt.ylim(0, 1.1)
plt.xlabel('Threshold')
plt.ylabel('Precision/ Recall')
plt.title('Precision and recall for different threshold values')
plt.legend(loc = 'upper right')

# %% 
# F1 score curve
plt.figure(figsize = (12, 6))
plt.plot(threshold, f1_score[1: ], label = "F1_score", linewidth = 3, color = 'green')
plt.scatter(threshold[f1_score[1: ] == max_f1], max_f1, label = 'Max F1 score = %0.3f' % (max_f1), s = 50, color = 'red')
plt.axvline(best_threshold, color = 'black', ls = '--', label = 'Threshold = %0.3f' % (best_threshold))
plt.axhline(max_f1, color = 'black', ls = '-')
plt.ylim(0, 1.1)
plt.xlabel('Threshold')
plt.ylabel('F1 score')
plt.title('F1 score for different threshold values')
plt.legend(loc = 'upper right')
plt.show()
print('Best threshold = %f' % (best_threshold))
print('Max F1 score = %f' % (max_f1))

# %% 
# Recall - Precision curve
plt.figure(figsize = (12, 6))
f_scores = np.linspace(0.2, 0.8, num = 4)

for f_score in f_scores:
    x = np.linspace(0.001, 1)
    y = f_score * x / (2 * x - f_score)
    plt.plot(x[y >= 0], y[y >= 0], color = 'gray', alpha = 0.2)
    plt.annotate('F1 = {0:0.2f}'.format(f_score), xy = (0.95, y[45] + 0.02))

plt.plot(recall[1: ], precision[1: ], label = 'Area = %0.3f' % (average_precision), linewidth = 3)
plt.scatter(recall[f1_score == max_f1], precision[f1_score == max_f1], label = 'F1 score = %0.3f' % (max_f1), s = 50, color = 'red')
plt.axvline(recall[f1_score == max_f1], color = 'black', ls = '--', label = 'Recall = %0.3f' % (recall[f1_score == max_f1]))
plt.axhline(precision[f1_score == max_f1], color = 'black', ls = '-', label = 'Precision = %0.3f' % (precision[f1_score == max_f1]))
plt.ylim(0, 1.1)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision - Recall curve')
plt.legend(loc = 'upper right')
plt.show()

# %% 
# Training Set
plt.figure(figsize = (12, 5))
plt.scatter(train_score.index[train_score['True_class'] == 0], 
            train_score[train_score['True_class'] == 0]['score'], 
            s = 5, label = 'Normal')
plt.scatter(train_score.index[train_score['True_class'] == 1], 
            train_score[train_score['True_class'] == 1]['score'], 
            s = 5, label = 'Exit the Bank')
plt.axhline(best_threshold, color = 'red', label = 'Threshold = %0.3f' % (best_threshold))
plt.xlabel('Index')
plt.ylabel('Score')
plt.title('Training Set')
plt.legend()
plt.show()
print('Best threshold = %f' % (best_threshold))

# %% 
# Predicting the Test set
y_test_score = classifier.predict_proba(X_test)[:, 1]
y_pred = y_test_score > best_threshold

# %% 
# Making the confusion matrix
cm = confusion_matrix(y_test, y_pred)
print(classification_report(y_test, y_pred))

plt.figure()
sns.heatmap(cm, cmap = "coolwarm", annot = True, linewidths = 0.5)
plt.title("Confusion Matrix")
plt.xlabel("Predicted class")
plt.ylabel("Real class")
plt.show()

# %% 
# ROC curve
fpr, tpr, thresholds = roc_curve(train_score.True_class, train_score.score)
roc_auc = auc(fpr, tpr)

plt.figure(figsize = (8, 5))
plt.plot(fpr, tpr, linewidth = 3, label = 'AUC = %0.3f' % (roc_auc))
plt.plot([0, 1], [0, 1], linewidth = 3)
plt.xlim(left = -0.02, right = 1)
plt.ylim(bottom = 0, top = 1.02)
plt.xlabel('False Positive Rate (FPR)')
plt.ylabel('True Positive Rate (TPR)')
plt.title('Receiver operating characteristic curve (ROC)')
plt.legend()
plt.show()
# %%
