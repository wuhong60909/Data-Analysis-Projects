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

# %%
# Fitting the classifier to the Training set: Logistic Regression
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from catboost import CatBoostClassifier

classifier = LogisticRegression(solver = 'liblinear', random_state = 0)
# classifier = LinearDiscriminantAnalysis()
# classifier = QuadraticDiscriminantAnalysis()
# classifier = SVC(kernel = 'linear', random_state = 0)
# classifier = GaussianNB()
# classifier = CatBoostClassifier(iterations = 100, learning_rate = 1, verbose = 0)

# %%
classifier.fit(X_train, y_train)
y_score = classifier.predict_proba(X_test)[:, 1]
# y_score = classifier.predict_log_proba(X_test)[:, 1]
# y_score = classifier.decision_function(X_test)

# %%
# Plot scores, ROC, PRF, PR curves
from curves import curves
a = curves(y_true = y_test, y_score = y_score)
a.scores_plot()
a.ROC_plot()
a.PRF_plot()
a.PR_plot()

# %%
# Choose thresholds by StratifiedKfold cross validation.
from selectThreshold import selectThresholdByCV
n_splits = 5
thresholds = np.linspace(0.001, 0.99, 10)
classifier = LogisticRegression(solver = 'liblinear', random_state = 0)
# %%
best_threshold, results = selectThresholdByCV(clf = classifier, X = X_train, y = y_train, 
                                              thresholds = thresholds, n_splits = n_splits, 
                                              plot_result = True)
best_threshold

# %%
classifier = LogisticRegression(solver = 'liblinear', random_state = 0)
classifier.fit(X_train, y_train)

y_test_score = classifier.predict_proba(X_test)[:, 1]
y_pred = (y_test_score >= 0.111)
print(classification_report(y_test, y_pred))
print(f1_score(y_test, y_pred))

# %%
# Applying Grid Search to find the best model and the best parameters
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold

classifier = LogisticRegression(solver = 'liblinear', random_state = 0)
parameter = [{'penalty': ['l1'], 'C': np.arange(0.01, 1, 0.5)}, 
             {'penalty': ['l2'], 'C': np.arange(0.01, 1, 0.5)}]

skf = StratifiedKFold(n_splits = 5, random_state = 0)
grid_search = GridSearchCV(estimator = classifier, 
                           param_grid = parameter,
                           scoring = 'average_precision',
                           cv = skf.split(X_train, y_train),
                           n_jobs = -1)
grid_search = grid_search.fit(X_train, y_train)
best_score = grid_search.best_score_
best_parameters = grid_search.best_params_
results = grid_search.cv_results_
best_parameters
# %%