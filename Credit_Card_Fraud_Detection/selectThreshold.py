# Importing Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import KFold, StratifiedKFold

# %%
def cross_val(clf, X, y, thresholds, cv):
    F1_array = []
    Precision_array = []    
    Recall_array = []
    Accuracy_array = []
    i = 0
    for train_ind, val_ind in cv.split(X, y):        
        print('Split{}'.format(i))
        i = i + 1
        F1 = []
        Precision = []
        Recall = []
        Accuracy = []
        X_train = X[train_ind]
        y_train = y[train_ind]        
        X_valid = X[val_ind]
        y_valid = y[val_ind]       
        
        # Output scores
        clf.fit(X_train, y_train)
        y_valid_score = clf.predict_proba(X_valid)[:, 1]
        
        # Test each thresholds and compute F1, Precisio, Recall, Accuracy
        for threshold in thresholds:
            y_pred = (y_valid_score > threshold)
            F1.append(f1_score(y_valid, y_pred, average = 'binary'))
            Precision.append(precision_score(y_valid, y_pred, average = 'binary'))
            Recall.append(recall_score(y_valid, y_pred, average = 'binary'))
            Accuracy.append(accuracy_score(y_valid, y_pred))
        
        # Collect scores of each thresholds
        F1_array.append(F1)
        Precision_array.append(Precision)
        Recall_array.append(Recall)
        Accuracy_array.append(Accuracy)
    
    # Collect scores of each splits
    F1_array = np.array(F1_array)
    Precision_array = np.array(Precision_array)
    Recall_array = np.array(Recall_array)
    Accuracy_array = np.array(Accuracy_array)

    return F1_array, Precision_array, Recall_array, Accuracy_array

# %%
def searchThreshold(F1, Precision, Recall, Accuracy, Thresholds, scoring): 
    cv_results = {'F1': F1, 
                  'Precision': Precision, 
                  'Recall': Recall, 
                  'Accuracy': Accuracy}

    # F1, Precision, Recall, Accuracy test score mean
    f1_mean_test_score = F1.mean(axis = 0)
    precision_mean_test_score = Precision.mean(axis = 0)
    recall_mean_test_score = Recall.mean(axis = 0)
    accuracy_mean_test_score = Accuracy.mean(axis = 0)

    # F1, Precision, Recall, Accuracy test score std
    f1_std_test_score = F1.std(axis = 0)
    precision_std_test_score = Precision.std(axis = 0)
    recall_std_test_score = Recall.std(axis = 0)
    accuracy_std_test_score = Accuracy.std(axis = 0)

    if scoring == 'accuracy':
        # Find the maximun of accuracy
        best_index = np.where(accuracy_mean_test_score == accuracy_mean_test_score.max())
    elif scoring == 'f1':
        # Find the maximun of f1 score
        best_index = np.where(f1_mean_test_score == f1_mean_test_score.max())
    elif scoring == 'precision':
        # Find the maximun of precision
        best_index = np.where(precision_mean_test_score == precision_mean_test_score.max())
    elif scoring == 'recall':
        # Find the maximun of recall
        best_index = np.where(recall_mean_test_score == recall_mean_test_score.max())
    
    # Find the best threshold
    best_threshold = Thresholds[best_index]

    # Find the threshold which maximizes the f1 score
    best_f1 = f1_mean_test_score[best_index]
    best_precision = precision_mean_test_score[best_index]
    best_recall = recall_mean_test_score[best_index]
    best_accuracy = accuracy_mean_test_score[best_index]

    # The corresponding test score std
    best_f1_std = f1_std_test_score[best_index]
    best_precision_std = precision_std_test_score[best_index]
    best_recall_std = recall_std_test_score[best_index]
    best_accuracy_std = accuracy_std_test_score[best_index]
    
    # Collect the results
    # mean_test_score
    mean_test_score = {'F1': f1_mean_test_score, 
                       'Precision': precision_mean_test_score, 
                       'Recall': recall_mean_test_score, 
                       'Accuracy': accuracy_mean_test_score}
    # std_test_score
    std_test_score = {'F1': f1_std_test_score, 
                      'Precision': precision_std_test_score, 
                      'Recall': recall_std_test_score, 
                      'Accuracy': accuracy_std_test_score}
    # best_score
    best_score = {'F1': best_f1, 
                  'Precision': best_precision, 
                  'Recall': best_recall, 
                  'Accuracy': best_accuracy}
    # best_std
    best_std = {'F1': best_f1_std, 
                'Precision': best_precision_std, 
                'Recall': best_recall_std, 
                'Accuracy': best_accuracy_std}
    
    return best_threshold, best_score, best_std, mean_test_score, std_test_score, cv_results

# %%
def selectThresholdByCV(clf, X, y, thresholds, cv = StratifiedKFold(n_splits = 5), scoring = 'accuracy', plot_result = True): 
    n_splits = cv.n_splits
    f1, precision, recall, accuracy = cross_val(clf = clf, 
                                                X = X, y = y, 
                                                thresholds = thresholds, 
                                                cv = cv)
    best_threshold, best_score, best_std, mean_test_score, std_test_score, cv_results = \
    searchThreshold(F1 = f1, 
                    Precision = precision, 
                    Recall = recall, 
                    Accuracy = accuracy, 
                    Thresholds = thresholds, 
                    scoring = scoring)
    if plot_result:
        # F1 scores vs thresholds
        plt.figure(figsize = (12, 5))
        plt.xlim((np.min(thresholds) - 0.05, np.max(thresholds) + 0.05))
        plt.ylim((0, 1.1))
        plt.xlabel('Thresholds')
        plt.ylabel('F1 CV')
        plt.title('{}-Fold Cross-Validated F1 scores'.format(n_splits))
        for ii in range(len(thresholds)):
            plt.plot(np.repeat(thresholds[ii], 2), [f1[:, ii].min(), f1[:, ii].max()], color = 'grey')
            plt.scatter(np.repeat(thresholds[ii], n_splits), f1[:, ii], 
                        marker = '+', color = 'grey')
        plt.scatter(thresholds, mean_test_score['F1'], s = 30, color = 'red')
        plt.scatter(best_threshold, best_score['F1'], 
                    s = 80, facecolors = 'none', edgecolors = 'g', 
                    label = 'F1 mean_test_score = %.3f (+/- %.3f)' % (best_score['F1'], 2 * best_std['F1']))
        plt.axvline(best_threshold, linestyle = ':', color = 'g', 
                    label = 'Best threshold = %.3f' % (best_threshold))
        plt.legend()
        plt.savefig('f1_cv.png')
        plt.show()

        # Precision vs thresholds
        plt.figure(figsize = (12, 5))
        plt.xlim((np.min(thresholds) - 0.05, np.max(thresholds) + 0.05))
        plt.ylim((0, 1.1))
        plt.xlabel('Thresholds')
        plt.ylabel('Precision CV')
        plt.title('{}-Fold Cross-Validated Precision'.format(n_splits))
        for ii in range(len(thresholds)):
            plt.plot(np.repeat(thresholds[ii], 2), [precision[:, ii].min(), precision[:, ii].max()], color = 'grey')
            plt.scatter(np.repeat(thresholds[ii], n_splits), precision[:, ii], 
                        marker = '+', color = 'grey')
        plt.scatter(thresholds, mean_test_score['Precision'], s = 30, color = 'red')
        plt.scatter(best_threshold, best_score['Precision'], 
                    s = 80, facecolors = 'none', edgecolors = 'g', 
                    label = 'Precision mean_test_score = %.3f (+/- %.3f)' % (best_score['Precision'], 2 * best_std['Precision']))
        plt.axvline(best_threshold, linestyle = ':', color = 'g', 
                    label = 'Best threshold = %.3f' % (best_threshold))
        plt.legend()
        plt.savefig('precision_cv.png')
        plt.show()

        # Recall vs thresholds
        plt.figure(figsize = (12, 5))
        plt.xlim((np.min(thresholds) - 0.05, np.max(thresholds) + 0.05))
        plt.ylim((0, 1.1))
        plt.xlabel('Thresholds')
        plt.ylabel('Recall CV')
        plt.title('{}-Fold Cross-Validated Recall'.format(n_splits))
        for ii in range(len(thresholds)):
            plt.plot(np.repeat(thresholds[ii], 2), [recall[:, ii].min(), recall[:, ii].max()], color = 'grey')
            plt.scatter(np.repeat(thresholds[ii], n_splits), recall[:, ii], 
                        marker = '+', color = 'grey')
        plt.scatter(thresholds, mean_test_score['Recall'], s = 30, color = 'red')
        plt.scatter(best_threshold, best_score['Recall'], 
                    s = 80, facecolors = 'none', edgecolors = 'g', 
                    label = 'Recall mean_test_score = %.3f (+/- %.3f)' % (best_score['Recall'], 2 * best_std['Recall']))
        plt.axvline(best_threshold, linestyle = ':', color = 'g', 
                    label = 'Best threshold = %.3f' % (best_threshold))
        plt.legend()
        plt.savefig('recall_cv.png')
        plt.show()

        # Accuracy vs thresholds
        plt.figure(figsize = (12, 5))
        plt.xlim((np.min(thresholds) - 0.05, np.max(thresholds) + 0.05))
        # plt.ylim((0.9, 1.1))
        plt.xlabel('Thresholds')
        plt.ylabel('Accuracy CV')
        plt.title('{}-Fold Cross-Validated Accuracy'.format(n_splits))
        for ii in range(len(thresholds)):
            plt.plot(np.repeat(thresholds[ii], 2), [accuracy[:, ii].min(), accuracy[:, ii].max()], color = 'grey')
            plt.scatter(np.repeat(thresholds[ii], n_splits), accuracy[:, ii], 
                        marker = '+', color = 'grey')
        plt.scatter(thresholds, mean_test_score['Accuracy'], s = 30, color = 'red')
        plt.scatter(best_threshold, best_score['Accuracy'], 
                    s = 80, facecolors = 'none', edgecolors = 'g', 
                    label = 'Accuracy mean_test_score = %.3f (+/- %.3f)' % (best_score['Accuracy'], 2 * best_std['Accuracy']))
        plt.axvline(best_threshold, linestyle = ':', color = 'g', 
                    label = 'Best threshold = %.3f' % (best_threshold))
        plt.legend()
        plt.savefig('accuracy_cv.png')
        plt.show()
    
    return best_threshold, best_score, best_std, mean_test_score, std_test_score, cv_results

# %%
