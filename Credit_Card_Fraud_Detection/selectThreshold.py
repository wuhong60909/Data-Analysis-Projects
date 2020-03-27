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

def cross_val(clf, X, y, thresholds, cv):    
    F1_array = []
    Precision_array = []    
    Recall_array = []
    kf = cv
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
        for threshold in thresholds:             
            y_pred = (y_valid_score >= threshold)
            F1 = f1_score(y_valid, y_pred, average = 'binary')
            Precision = precision_score(y_valid, y_pred, average = 'binary')
            Recall = recall_score(y_valid, y_pred, average = 'binary')
            a1.append(F1)
            a2.append(Precision)
            a3.append(Recall)
        F1_array.append(a1)
        Precision_array.append(a2)
        Recall_array.append(a3)
    F1_array = np.array(F1_array)
    Precision_array = np.array(Precision_array)
    Recall_array = np.array(Recall_array)
    return F1_array, Precision_array, Recall_array

def outputThreshold(F1, Precision, Recall, Thresholds): 
    n_splits = F1.shape[0]
    cv_results = {'F1': F1, 'Precision': Precision, 'Recall': Recall}

    # F1, Precision, Recall mean test score
    f1_mean_test_score = F1.mean(axis = 0)
    precision_mean_test_score = Precision.mean(axis = 0)
    recall_mean_test_score = Recall.mean(axis = 0)

    # Find the maximun of f1 score
    best_index = np.where(f1_mean_test_score == f1_mean_test_score.max())
    
    # Find the threshold which maximizes the f1 score
    best_f1 = f1_mean_test_score[best_index]
    best_precision = precision_mean_test_score[best_index]
    best_recall = recall_mean_test_score[best_index]
    best_threshold = Thresholds[best_index]
    
    # Collect the results
    best = {'F1': best_f1, 
            'Precision': best_precision, 
            'Recall': best_recall, 
            'Threshold': best_threshold}
    
    # Figures vs thresholds
    # F1 scores vs thresholds
    plt.figure(figsize = (12, 5))
    plt.xlim((np.min(Thresholds) - 0.05, np.max(Thresholds) + 0.05))
    plt.ylim((0, 1.1))
    plt.xlabel('Thresholds')
    plt.ylabel('F1 Scores')
    plt.title('{}-Fold Cross-Validated F1 scores'.format(n_splits))
    for ii in range(len(Thresholds)):
        plt.plot(np.repeat(Thresholds[ii], 2), [F1[:, ii].min(), F1[:, ii].max()], color = 'grey')
        plt.scatter(np.repeat(Thresholds[ii], n_splits), F1[:, ii], 
                    marker = '+', color = 'grey')
    plt.scatter(Thresholds, f1_mean_test_score, s = 30, color = 'red')
    plt.scatter(best_threshold, best_f1, 
                s = 80, facecolors = 'none', edgecolors = 'g', 
                label = 'F1 mean_test_score = %.3f' % (best_f1))
    plt.axvline(best_threshold, linestyle = ':', color = 'g', 
                label = 'Best threshold = %.3f' % (best_threshold))
    plt.legend()
    plt.show()
    
    # Precision vs thresholds
    plt.figure(figsize = (12, 5))
    plt.xlim((np.min(Thresholds) - 0.05, np.max(Thresholds) + 0.05))
    plt.ylim((0, 1.1))
    plt.xlabel('Thresholds')
    plt.ylabel('Precision')
    plt.title('{}-Fold Cross-Validated Precision'.format(n_splits))
    for ii in range(len(Thresholds)):
        plt.plot(np.repeat(Thresholds[ii], 2), [Precision[:, ii].min(), Precision[:, ii].max()], color = 'grey')
        plt.scatter(np.repeat(Thresholds[ii], n_splits), Precision[:, ii], 
                    marker = '+', color = 'grey')
    plt.scatter(Thresholds, precision_mean_test_score, s = 30, color = 'red')
    plt.scatter(best_threshold, best_precision, 
                s = 80, facecolors = 'none', edgecolors = 'g', 
                label = 'Precision mean_test_score = %.3f' % (best_precision))
    plt.axvline(best_threshold, linestyle = ':', color = 'g', 
                label = 'Best threshold = %.3f' % (best_threshold))
    plt.legend()
    plt.show()
    
    # Recall vs thresholds
    plt.figure(figsize = (12, 5))
    plt.xlim((np.min(Thresholds) - 0.05, np.max(Thresholds) + 0.05))
    plt.ylim((0, 1.1))
    plt.xlabel('Thresholds')
    plt.ylabel('Recall')
    plt.title('{}-Fold Cross-Validated Recall'.format(n_splits))
    for ii in range(len(Thresholds)):
        plt.plot(np.repeat(Thresholds[ii], 2), [Recall[:, ii].min(), Recall[:, ii].max()], color = 'grey')
        plt.scatter(np.repeat(Thresholds[ii], n_splits), Recall[:, ii], 
                    marker = '+', color = 'grey')
    plt.scatter(Thresholds, recall_mean_test_score, s = 30, color = 'red')
    plt.scatter(best_threshold, best_recall, 
                s = 80, facecolors = 'none', edgecolors = 'g', 
                label = 'Recall mean_test_score = %.3f' % (best_recall))
    plt.axvline(best_threshold, linestyle = ':', color = 'g', 
                label = 'Best threshold = %.3f' % (best_threshold))
    plt.legend()
    plt.show()    
    return best, cv_results

def selectThresholdByCV(clf, X, y, thresholds, cv = StratifiedKFold(n_splits = 5), plot_result = True): 
    f1, precision, recall = cross_val(clf = clf, X = X, y = y, 
                                      thresholds = thresholds, cv = cv)
    return outputThreshold(F1 = f1, Precision = precision, Recall = recall, Thresholds = thresholds)
