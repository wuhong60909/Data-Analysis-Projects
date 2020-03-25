# -*- coding: utf-8 -*-
"""
Created on Mon Mar 23 13:30:11 2020

@author: Hong
"""

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

# %%
# Importing dataset
dataset = pd.read_csv('creditcard.csv')
dataset_X = dataset.iloc[:, 0:30]
dataset_y = dataset.iloc[:, 30]

# %%
dataset.info()

# %%
dataset.describe()

# %%
dataset.isnull().sum()

# %%
dataset.head()

# %%
## Select the features
v_features = dataset.iloc[:, 0:30].columns

plt.figure(figsize = (12, 120))
gs = gridspec.GridSpec(30, 1)
for i, cn in enumerate(dataset[v_features]):
    ax = plt.subplot(gs[i])    
    sns.distplot(dataset[cn][dataset.Class == 0], bins = 50, label = 'Normal')
    sns.distplot(dataset[cn][dataset.Class == 1], bins = 50, label = 'Fraud')
    ax.set_xlabel('')
    ax.set_title('histogram of feature: ' + str(cn))
    plt.legend()
plt.show()

# %%
X = dataset_X.values
X = X[:, 1:] # Drop 'Time'
y = dataset_y.values

# %%
# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, shuffle = False)

# %%
# Feature Scaling: Only 'Amount'
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train[:, -1:] = sc.fit_transform(X_train[:, -1:])
X_test[:, -1:] = sc.transform(X_test[:, -1:])

# %%
# Fitting the classifier to the Training set: Logistic Regression
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(solver = 'liblinear', random_state = 0)
classifier.fit(X_train, y_train)

# Predict Test set
y_pred = classifier.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
print(cm)
print('accuracy = {}'.format(accuracy_score(y_test, y_pred)))
print('precision = {}'.format(precision_score(y_test, y_pred)))
print('recall = {}'.format(recall_score(y_test, y_pred)))
print('f1 score = {}'.format(f1_score(y_test, y_pred)))
print(classification_report(y_test, y_pred))

sns.heatmap(cm, cmap = "coolwarm", annot = True, linewidths = 0.5)
plt.title("Confusion Matrix")
plt.xlabel("Predicted class")
plt.ylabel("Real class")
plt.show()

# %%
# Training set
train_score = pd.DataFrame({'score': classifier.predict_proba(X_train)[:, 1], 
                            'True_class': y_train})
print(train_score.describe())
# %%
plt.figure(figsize = (12, 5))
plt.scatter(train_score.index[train_score['True_class'] == 0], 
            train_score[train_score['True_class'] == 0]['score'], 
            s = 5, label = 'Normal')
plt.scatter(train_score.index[train_score['True_class'] == 1], 
            train_score[train_score['True_class'] == 1]['score'], 
            s = 5, label = 'Frauld')
plt.xlabel('Index')
plt.ylabel('Score')
plt.title('Training Set')
plt.legend()
plt.show()

# %%
# Test set =================================================================
test_score = pd.DataFrame({'score': classifier.predict_proba(X_test)[:, 1], 
                           'True_class': y_test})
print(test_score.describe())

# %% 
plt.figure(figsize = (12, 5))
plt.scatter(test_score.index[test_score['True_class'] == 0], 
            test_score[test_score['True_class'] == 0]['score'], 
            s = 5, label = 'Normal')
plt.scatter(test_score.index[test_score['True_class'] == 1], 
            test_score[test_score['True_class'] == 1]['score'], 
            s = 5, label = 'Frauld')
plt.xlabel('Index')
plt.ylabel('Score')
plt.title('Test Set')
plt.legend()
plt.show()

# %%
# ROC curve
fpr, tpr, thresholds = roc_curve(test_score.True_class, test_score.score)
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
# Plotting the precision recall curve.
precision, recall, threshold = precision_recall_curve(test_score.True_class, test_score.score)
average_precision = average_precision_score(test_score.True_class, test_score.score)
f1_scores = 2 * precision * recall / (precision + recall)

# %%
# Precision, Recall, F1 score curve 
plt.figure(figsize = (12, 6))
plt.plot(threshold, precision[1: ], label = "Precision", linewidth = 3)
plt.plot(threshold, recall[1: ], label ="Recall", linewidth = 3)
plt.plot(threshold, f1_scores[1: ], label = "F1_score", linewidth = 3, color = 'green')
plt.ylim(0, 1.1)
plt.xlabel('Threshold')
plt.ylabel('Precision/ Recall/ F1 score')
plt.title('Precision and recall for different threshold values')
plt.legend(loc = 'upper right')

# %%
print('Max F1 score = %.3f' % (f1_scores.max()))
print('Threshold = %.3f' % (threshold[f1_scores[1: ] == f1_scores.max()]))

# %% 
# Recall - Precision curve
plt.figure(figsize = (12, 6))
f_scores = np.linspace(0.2, 0.8, num = 4)
for f_score in f_scores:
    x = np.linspace(0.001, 1)
    y = f_score * x / (2 * x - f_score)
    plt.plot(x[y >= 0], y[y >= 0], color = 'gray', alpha = 0.2)
    plt.annotate('F1 = {0:0.2f}'.format(f_score), xy = (0.95, y[45] + 0.02))

plt.plot(recall[1: ], precision[1: ], 
         label = 'Average Precision(AP) = %0.3f' % (average_precision), 
         linewidth = 3)
plt.ylim(0, 1.1)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision - Recall curve')
plt.legend(loc = 'upper right')
plt.show()

# %%
# Predicting the Test set
best_threshold = 0.117
y_test_score = classifier.predict_proba(X_test)[:, 1]
y_pred2 = y_test_score >= best_threshold

# %%
cm = confusion_matrix(y_test, y_pred2)
print(cm)
print('accuracy = {}'.format(accuracy_score(y_test, y_pred2)))
print('precision = {}'.format(precision_score(y_test, y_pred2)))
print('recall = {}'.format(recall_score(y_test, y_pred2)))
print('F1 score = {}'.format(f1_score(y_test, y_pred2)))
print(classification_report(y_test, y_pred2))

sns.heatmap(cm, cmap = "coolwarm", annot = True, linewidths = 0.5)
plt.title("Confusion Matrix")
plt.xlabel("Predicted class")
plt.ylabel("Real class")
plt.show()

# %%
def cross_val(clf, X, y, thresholds, n_splits = 5, plot_result = True):    
    F1_array = []
    Precision_array = []    
    Recall_array = []    
    # kf = KFold(n_splits = n_splits, random_state = 0)
    kf = StratifiedKFold(n_splits = n_splits, random_state = 0)
    i = 0
    for train_ind, val_ind in kf.split(X, y):        
        i = i + 1        
        a1 = []
        a2 = []
        a3 = []        
        X_train = X[train_ind]
        y_train = y[train_ind]        
        X_valid = X[val_ind]
        y_valid = y[val_ind]
        clf.fit(X_train, y_train)
        
        # Output scores
        y_valid_score = clf.predict_proba(X_valid)[:, 1]        
        
        # Test each thresholds and compute F1, Precisio, Recall        
        print('\nSplit {} ========================================='.format(i))
        for threshold in thresholds:             
            y_pred = y_valid_score >= threshold
            F1 = f1_score(y_valid, y_pred, average = 'binary')
            Precision = precision_score(y_valid, y_pred, average = 'binary')
            Recall = recall_score(y_valid, y_pred, average = 'binary')            
            a1.append(F1)
            a2.append(Precision)
            a3.append(Recall)
            print('-----------------------------------')   
            print('Threshold = %.3f' % (threshold))
            print('-----------------------------------')   
            print('F1 score = %.3f' % (F1))
            print('Precision = %.3f' % (Precision))
            print('Recall = %.3f' % (Recall))
            
        F1_array.append(a1)
        Precision_array.append(a2)
        Recall_array.append(a3)
    
    F1_array = np.array(F1_array)
    Precision_array = np.array(Precision_array)
    Recall_array = np.array(Recall_array)
    
    return Precision_array, Recall_array, F1_array    

# %%
classifier = LogisticRegression(solver = 'liblinear', random_state = 0)
n_splits = 5
thresholds = np.linspace(0.001, 0.99, 100)
Precision, Recall, F1_scores = cross_val(clf = classifier, X = X_train, y = y_train, 
                                         thresholds = thresholds, n_splits = n_splits)

# %%
def selectThresholdByCV(F1, Precision, Recall, Thresholds):    
    n_splits = F1_scores.shape[0]    
    f1_thres = F1.mean(axis = 0)
    precision_thres = Precision.mean(axis = 0)
    recall_thres = Recall.mean(axis = 0)
    
    # Find the maximun of f1 score
    best_index = np.where(f1_thres == f1_thres.max())
    
    # Find the threshold which maximizes the f1 score
    best_f1 = f1_thres[best_index]
    best_precision = precision_thres[best_index]
    best_recall = recall_thres[best_index]
    best_threshold = Thresholds[best_index]
    
    # Figures vs thresholds
    # F1 scores vs thresholds
    plt.figure(figsize = (12, 5))
    plt.xlim((np.min(thresholds) - 0.05, np.max(thresholds) + 0.05))
    plt.ylim((0, 1.1))
    plt.xlabel('Thresholds')
    plt.ylabel('F1 Scores')
    plt.title('{}-Fold Cross-Validated F1 scores'.format(n_splits))
    for ii in range(len(Thresholds)):
        plt.plot(np.repeat(Thresholds[ii], 2), [F1[:, ii].min(), F1[:, ii].max()], color = 'grey')
        plt.scatter(np.repeat(Thresholds[ii], n_splits), F1[:, ii], 
                    marker = '+', color = 'grey')        
    # plt.plot(Thresholds, f1_thres)
    plt.scatter(Thresholds, f1_thres, s = 30, color = 'red')
    plt.scatter(best_threshold, best_f1, 
                s = 80, facecolors = 'none', edgecolors = 'g', 
                label = 'Best F1 scoreCV = %.3f' % (best_f1))
    plt.axvline(best_threshold, linestyle = ':', color = 'g', 
                label = 'Best threshold = %.3f' % (best_threshold))
    plt.legend()
    plt.show()
    
    # Precision vs thresholds
    plt.figure(figsize = (12, 5))
    plt.xlim((np.min(thresholds) - 0.05, np.max(thresholds) + 0.05))
    plt.ylim((0, 1.1))
    plt.xlabel('Thresholds')
    plt.ylabel('Precision')
    plt.title('{}-Fold Cross-Validated Precision'.format(n_splits))
    for ii in range(len(Thresholds)):
        plt.plot(np.repeat(Thresholds[ii], 2), [Precision[:, ii].min(), Precision[:, ii].max()], color = 'grey')
        plt.scatter(np.repeat(Thresholds[ii], n_splits), Precision[:, ii], 
                    marker = '+', color = 'grey')        
    # plt.plot(Thresholds, precision_thres)
    plt.scatter(Thresholds, precision_thres, s = 30, color = 'red')
    plt.scatter(best_threshold, best_precision, 
                s = 80, facecolors = 'none', edgecolors = 'g', 
                label = 'Best PrecisionCV = %.3f' % (best_precision))
    plt.axvline(best_threshold, linestyle = ':', color = 'g', 
                label = 'Best threshold = %.3f' % (best_threshold))
    plt.legend()
    plt.show()
    
    # Recall vs thresholds
    plt.figure(figsize = (12, 5))
    plt.xlim((np.min(thresholds) - 0.05, np.max(thresholds) + 0.05))
    plt.ylim((0, 1.1))
    plt.xlabel('Thresholds')
    plt.ylabel('Recall')
    plt.title('{}-Fold Cross-Validated Recall'.format(n_splits))
    for ii in range(len(Thresholds)):
        plt.plot(np.repeat(Thresholds[ii], 2), [Recall[:, ii].min(), Recall[:, ii].max()], color = 'grey')
        plt.scatter(np.repeat(Thresholds[ii], n_splits), Recall[:, ii], 
                    marker = '+', color = 'grey')        
    # plt.plot(Thresholds, recall_thres)
    plt.scatter(Thresholds, recall_thres, s = 30, color = 'red')
    plt.scatter(best_threshold, best_recall, 
                s = 80, facecolors = 'none', edgecolors = 'g', 
                label = 'Best RecallCV = %.3f' % (best_recall))
    plt.axvline(best_threshold, linestyle = ':', color = 'g', 
                label = 'Best threshold = %.3f' % (best_threshold))
    plt.legend()
    plt.show()
    
    return best_f1, best_precision, best_recall, best_threshold
    
# %%
best_f1, best_precision, best_recall, best_threshold = selectThresholdByCV(F1 = F1_scores, 
                                                                           Precision = Precision, 
                                                                           Recall = Recall, 
                                                                           Thresholds = thresholds)

# %%
print('Best threshold = %.3f' % (best_threshold))
print('F1 score CV = %.3f' % (best_f1))
print('Precision CV = %.3f' % (best_precision))
print('Recall CV = %.3f' % (best_recall))

# %%
# Predicting the Test set
classifier = LogisticRegression(solver = 'liblinear', random_state = 0)
classifier.fit(X_train, y_train)
y_test_score = classifier.predict_proba(X_test)[:, 1]
y_pred = (y_test_score >= best_threshold)
# %%
cm = confusion_matrix(y_test, y_pred)
print(cm)
print('accuracy = {}'.format(accuracy_score(y_test, y_pred)))
print('precision = {}'.format(precision_score(y_test, y_pred)))
print('recall = {}'.format(recall_score(y_test, y_pred)))
print('F1 score = {}'.format(f1_score(y_test, y_pred)))
print(classification_report(y_test, y_pred))

sns.heatmap(cm, cmap = "coolwarm", annot = True, linewidths = 0.5)
plt.title("Confusion Matrix")
plt.xlabel("Predicted class")
plt.ylabel("Real class")
plt.show()

# %%

# %%