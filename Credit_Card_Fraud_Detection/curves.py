import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, \
classification_report, confusion_matrix, precision_recall_curve, \
average_precision_score, roc_curve, auc

class curves:
    def __init__(self, clf, X, y):
        '''
        Binary label case only!
        '''
        self.clf = clf
        self.X = X
        self.y = y
        self.test_score = pd.DataFrame({'True_class': self.y, 
                                        'score': clf.predict_proba(self.X)[:, 1]})
        # ROC curve
        self.fpr, self.tpr, _ = roc_curve(self.test_score.True_class, self.test_score.score)
        self.roc_auc = auc(self.fpr, self.tpr)

        # Rrecision, Recall, F1 score curve.
        self.precision, self.recall, self.threshold = precision_recall_curve(self.test_score.True_class, self.test_score.score)
        self.average_precision = average_precision_score(self.test_score.True_class, self.test_score.score)
        self.f1_scores = 2 * self.precision * self.recall / (self.precision + self.recall)

    def scores_plot(self):
        # scatter plot scores
        plt.figure(figsize = (12, 5))
        plt.scatter(self.test_score.index[self.test_score['True_class'] == 0], 
                    self.test_score[self.test_score['True_class'] == 0]['score'], 
                    s = 5, label = '0')
        plt.scatter(self.test_score.index[self.test_score['True_class'] == 1], 
                    self.test_score[self.test_score['True_class'] == 1]['score'], 
                    s = 5, label = '1')
        plt.xlabel('Index')
        plt.ylabel('Score')
        plt.title('Classifier: ' + self.clf.__class__.__name__)
        plt.legend()
        plt.show()

    def ROC_plot(self):     
        plt.figure(figsize = (8, 5))
        plt.plot(self.fpr, self.tpr, linewidth = 3, label = 'AUC = %0.3f' % (self.roc_auc))
        plt.plot([0, 1], [0, 1], linewidth = 3)
        plt.xlim(left = -0.02, right = 1)
        plt.ylim(bottom = 0, top = 1.02)
        plt.xlabel('False Positive Rate (FPR)')
        plt.ylabel('True Positive Rate (TPR)')
        plt.title('Receiver operating characteristic curve ({})'.format(self.clf.__class__.__name__))
        plt.legend()
        plt.show()

    def PRF_plot(self):
        # Precision, Recall, F1 score curve
        max_f1 = self.f1_scores.max()
        best_threshold = self.threshold[self.f1_scores[1: ] == max_f1]

        plt.figure(figsize = (12, 6))
        plt.plot(self.threshold, self.precision[1: ], label = "Precision", linewidth = 3)
        plt.plot(self.threshold, self.recall[1: ], label = "Recall", linewidth = 3)
        plt.plot(self.threshold, self.f1_scores[1: ], label = "F1 score", linewidth = 3, color = 'green')
        plt.axvline(best_threshold, color = 'black', ls = '--', label = 'Threshold = %0.3f' % (best_threshold))
        plt.axhline(max_f1, color = 'black', ls = '-', label = 'Max F1 score = %0.3f' % (max_f1))
        plt.ylim(0, 1.1)
        plt.xlabel('Threshold')
        plt.ylabel('Precision/ Recall/ F1 score')
        plt.title('Precision and recall for different threshold values ({})'.format(self.clf.__class__.__name__))
        plt.legend(loc = 'upper right')
        plt.show()

    def PR_plot(self):
        # Precision - Recall curve
        plt.figure(figsize = (12, 6))
        f_scores = np.linspace(0.2, 0.8, num = 4)
        for f_score in f_scores:
            x = np.linspace(0.001, 1)
            y = f_score * x / (2 * x - f_score)
            plt.plot(x[y >= 0], y[y >= 0], color = 'gray', alpha = 0.2)
            plt.annotate('F1 = {0:0.2f}'.format(f_score), xy = (0.95, y[45] + 0.02))

        plt.plot(self.recall[1: ], self.precision[1: ], 
                 label = 'Average Precision(AP) = %0.3f' % (self.average_precision), 
                 linewidth = 3)
        plt.ylim(0, 1.1)
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision - Recall curve ({})'.format(self.clf.__class__.__name__))
        plt.legend(loc = 'upper right')
        plt.show()
