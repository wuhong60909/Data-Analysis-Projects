# %% [markdown]
# ## Importing Libraries

# %%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import matplotlib.gridspec as gridspec
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, \
classification_report, confusion_matrix, precision_recall_curve, \
average_precision_score, roc_curve, auc
from sklearn.model_selection import KFold, StratifiedKFold

# %% [markdown]
# ## Importing dataset

# %%
dataset = pd.read_csv('creditcard.csv')

# %% [markdown]
# ## View information about the dataset.
# %%
dataset.info()

# %%
dataset.describe()

# %%
dataset.isnull().sum()

# %%
dataset.head()

# %% [markdown]
# ## Data Visualization

# %%
# Select the features
v_features = dataset.iloc[:, 0:30].columns

plt.figure(figsize = (12, 120))
gs = gridspec.GridSpec(30, 1)
for i, cn in enumerate(dataset[v_features]):
    ax = plt.subplot(gs[i])    
    sns.distplot(dataset[cn][dataset.Class == 0], bins = 50, label = 'Normal')
    sns.distplot(dataset[cn][dataset.Class == 1], bins = 50, label = 'Fraud')
    ax.set_xlabel('')
    ax.set_title('Histogram of feature: ' + str(cn))
    plt.legend()
plt.show()

# %% [markdown]
# ## Collect columns we need
# %%
dataset_X = dataset.drop(['Class'], axis = 1)
dataset_y = dataset['Class']
y = dataset_y.values
# %%
# Drop 'Time', 'Amount'
drop_list = ['Time', 'Amount']
X = dataset_X.drop(drop_list, axis = 1)
X = X.values

# %% [markdown]
# ## Splitting the dataset into the Training set and Test set
# %%
def split_data(X, y):
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, stratify = y, random_state = 0)
    print('train-set size: ', len(y_train), 
          '\ntest-set size: ', len(y_test))    
    print('fraud cases in train-set', sum(y_train),
          '\nfraud cases in test-set: ', sum(y_test))
    return X_train, X_test, y_train, y_test

X_train, X_test, y_train, y_test = split_data(X, y)

# %% [markdown]
# ## Define function to get predictions
# %%
def get_predictions(y_true, y_pred): 
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, cmap = "coolwarm", annot = True, linewidths = 0.5)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted class")
    plt.ylabel("Real class")
    plt.show()
    print('==============================')
    print(cm)
    print('==============================')
    print('accuracy = {}'.format(accuracy_score(y_true, y_pred)))
    print('precision = {}'.format(precision_score(y_true, y_pred)))
    print('recall = {}'.format(recall_score(y_true, y_pred)))
    print('f1 score = {}'.format(f1_score(y_true, y_pred)))
    print('==============================')
    print(classification_report(y_true, y_pred))

# %% [marldown]
# ## Fitting the classifier to the Training set: Logistic Regression
# %%
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(solver = 'liblinear', random_state = 0)
classifier.fit(X_train, y_train)

# Predict Test set
y_pred = classifier.predict(X_test)
y_score = classifier.predict_proba(X_test)[:, 1]
get_predictions(y_true = y_test, y_pred = y_pred)
# %% [markdown]
# ## Plot scores, ROC, PRF, PR curves
# %%
import curves as C
C.plot_scores(y_true = y_test, y_score = y_score)
fpr, tpr = C.plot_ROC(y_true = y_test, y_score = y_score)
precision, recall, threshold, f1_scores = C.plot_precision_recall_vs_threshold(y_true = y_test, y_score = y_score)
C.plot_precision_recall(y_true = y_test, y_score = y_score)


# %%
# Predicting the Test set
threshold = 0.1
y_score = classifier.predict_proba(X_test)[:, 1]
y_pred = (y_score > threshold)
get_predictions(y_true = y_test, y_pred = y_pred)

# %% [markdown]
# ## Select threshold by stratifiefKfold cross-validation
# %%
from selectThreshold import selectThresholdByCV
skf = StratifiedKFold(n_splits = 10)
thresholds = np.array([0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
classifier = LogisticRegression(solver = 'liblinear', random_state = 0)
# %%
best_threshold, results = selectThresholdByCV(clf = classifier, X = X_train, y = y_train, 
                                              thresholds = thresholds, cv = skf, 
                                              plot_result = True)
best_threshold

# %% [markdown]
# Predicting the Test set using best_threshold
# %%
classifier = LogisticRegression(solver = 'liblinear', random_state = 0)
classifier.fit(X_train, y_train)
threshold = best_threshold['Threshold']
y_score = classifier.predict_proba(X_test)[:, 1]
y_pred = (y_score > threshold)
get_predictions(y_true = y_test, y_pred = y_pred)

# %%


# %%
