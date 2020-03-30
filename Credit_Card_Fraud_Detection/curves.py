import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import precision_recall_curve, average_precision_score, roc_curve, roc_auc_score

def plot_scores(y_true, y_score):
    test_score = pd.DataFrame({'True_class': y_true, 
                               'score': y_score})
    plt.figure(figsize = (12, 8))
    plt.scatter(test_score.index[test_score['True_class'] == 0], 
                test_score[test_score['True_class'] == 0]['score'], 
                s = 5, label = '0')
    plt.scatter(test_score.index[test_score['True_class'] == 1], 
                test_score[test_score['True_class'] == 1]['score'], 
                s = 5, label = '1')
    plt.xlabel('Index')
    plt.ylabel('Score')
    plt.title('Score scatter plot')
    plt.legend()
    plt.show()

def plot_ROC(y_true, y_score): 
    test_score = pd.DataFrame({'True_class': y_true, 
                               'score': y_score})
    # ROC curve
    fpr, tpr, _ = roc_curve(test_score.True_class, test_score.score)
    roc_auc = roc_auc_score(test_score.True_class, test_score.score)
    
    plt.figure(figsize = (12, 8))
    plt.plot(fpr, tpr, linewidth = 3, label = 'AUC = %0.3f' % (roc_auc))
    plt.plot([0, 1], [0, 1], '--k', linewidth = 3)
    plt.xlim(left = -0.02, right = 1)
    plt.ylim(bottom = 0, top = 1.02)
    plt.xlabel('False Positive Rate (FPR)')
    plt.ylabel('True Positive Rate (TPR)')
    plt.title('Receiver operating characteristic curve (ROC)')
    plt.legend(loc = 'lower right')
    plt.show()
    return fpr, tpr, roc_auc

def plot_precision_recall_vs_threshold(y_true, y_score):     
    test_score = pd.DataFrame({'True_class': y_true, 
                               'score': y_score})
    # Rrecision, Recall, F1 score curve.
    precision, recall, threshold = precision_recall_curve(test_score.True_class, test_score.score)
    f1_scores = 2 * precision * recall / (precision + recall)

    plt.figure(figsize = (12, 8))
    plt.plot(threshold, precision[1: ], label = "Precision", linewidth = 3)
    plt.plot(threshold, recall[1: ], label = "Recall", linewidth = 3)
    plt.plot(threshold, f1_scores[1: ], label = "F1 score", linewidth = 3, color = 'green')
    plt.ylim(0, 1.1)
    plt.xlabel('Threshold')
    plt.ylabel('Precision/ Recall/ F1 score')
    plt.title('Precision, recall and F1 score for different threshold values')
    plt.legend(loc = 'upper right')
    plt.show()
    return precision, recall, threshold, f1_scores

def plot_precision_recall(y_true, y_score):
    test_score = pd.DataFrame({'True_class': y_true, 
                               'score': y_score})
    # Precision - Recall curve
    precision, recall, _ = precision_recall_curve(test_score.True_class, test_score.score)
    average_precision = average_precision_score(test_score.True_class, test_score.score)
        
    plt.figure(figsize = (12, 8))
    f_scores = np.linspace(0.2, 0.8, num = 7)
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
    return precision, recall, average_precision
