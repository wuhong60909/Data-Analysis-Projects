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
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
import xgboost as xgb
import curves as C

# %%
dataset = pd.read_csv('creditcard.csv')
dataset_X = dataset.drop(['Class'], axis = 1)
dataset_y = dataset['Class']

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

# %% [markdown]
# ## Feature importance plot
# %%
def plot_feature_importance(model, predictors):
    tmp = pd.DataFrame({'Feature': predictors, 'Feature importance': model.feature_importances_})
    tmp = tmp.sort_values(by = 'Feature importance', ascending = False)
    plt.figure(figsize = (15, 8))
    plt.title('Features importance', fontsize = 14)
    s = sns.barplot(x = 'Feature', y = 'Feature importance', data = tmp)
    s.set_xticklabels(s.get_xticklabels(), rotation = 45)
    plt.show()

# %% [markdown]
# ## case 1: Drop 'Time', 'Amount'
# %%
drop_list = ['Time', 'Amount']
# drop_list = ['Time', 'Amount', 'V28', 'V27', 'V26', 'V25', 'V24', 'V23', 'V22', 'V20', 'V15', 'V13', 'V8']

X = dataset_X.drop(drop_list, axis = 1)
predictor_name = X.columns
X = X.values
y = dataset_y.values
print(predictor_name)
X_train, X_test, y_train, y_test = split_data(X, y)

# %% [markdown]
# ## Fitting the classifier to the Training set: Logistic Regression
# %%
lr_clf = LogisticRegression(solver = 'liblinear', random_state = 0)
lr_clf.fit(X_train, y_train)
y_score = lr_clf.predict_proba(X_test)[:, 1]
y_pred = lr_clf.predict(X_test)
get_predictions(y_true = y_test, y_pred = y_pred)

# %%
# ## Plot scores, ROC, Precision, Recall, F1 curve, Precision-Recall curve
C.plot_scores(y_true = y_test, y_score = y_score)
fpr_lr, tpr_lr, roc_auc_lr = C.plot_ROC(y_true = y_test, y_score = y_score)
precision_lr, recall_lr, threshold_lr, f1_scores_lr = C.plot_precision_recall_vs_threshold(y_true = y_test, y_score = y_score)
_, _, pr_auc_lr = C.plot_precision_recall(y_true = y_test, y_score = y_score)


# %% [markdown]
# ##  Extreme Gradient Boosting (XGB)
# %%
xgb_clf = xgb.XGBClassifier(n_jobs = -1, n_estimators = 200)
xgb_clf.fit(X_train, y_train)
y_score = xgb_clf.predict_proba(X_test)[:,1]
y_pred = xgb_clf.predict(X_test)
get_predictions(y_true = y_test, y_pred = y_pred)

# %%
# ## Plot scores, ROC, Precision, Recall, F1 curve, Precision-Recall curve
C.plot_scores(y_true = y_test, y_score = y_score)
fpr_xgb, tpr_xgb, roc_auc_xgb = C.plot_ROC(y_true = y_test, y_score = y_score)
precision_xgb, recall_xgb, threshold_xgb, f1_scores_xgb = C.plot_precision_recall_vs_threshold(y_true = y_test, y_score = y_score)
_, _, pr_auc_xgb = C.plot_precision_recall(y_true = y_test, y_score = y_score)

# %%
plot_feature_importance(model = xgb_clf, predictors = predictor_name)

# %%
# Ramdom forest Classifier
rf_clf = RandomForestClassifier(n_estimators = 200, 
                                max_features = 3, 
                                min_samples_leaf = 1, 
                                min_samples_split = 2, 
                                n_jobs = -1,
                                random_state = 0)

rf_clf.fit(X_train, y_train)
y_score = rf_clf.predict_proba(X_test)[:,1]
y_pred = rf_clf.predict(X_test)
get_predictions(y_true = y_test, y_pred = y_pred)

# %%
# ## Plot scores, ROC, Precision, Recall, F1 curve, Precision-Recall curve
C.plot_scores(y_true = y_test, y_score = y_score)
fpr_rf, tpr_rf, roc_auc_rf = C.plot_ROC(y_true = y_test, y_score = y_score)
precision_rf, recall_rf, threshold_rf, f1_scores_rf = C.plot_precision_recall_vs_threshold(y_true = y_test, y_score = y_score)
_, _, pr_auc_rf = C.plot_precision_recall(y_true = y_test, y_score = y_score)

# %%
plot_feature_importance(model = rf_clf, predictors = predictor_name)

# %% [markdown]
# ## Voting Classifier
# %%
lr_clf = LogisticRegression(solver = 'liblinear', random_state = 0)
xgb_clf = xgb.XGBClassifier(n_jobs = -1, n_estimators = 200)
rf_clf = RandomForestClassifier(n_estimators = 200, 
                                max_features = 3, 
                                min_samples_leaf = 1, 
                                min_samples_split = 2, 
                                n_jobs = -1,
                                random_state = 0)
voting_clf = VotingClassifier (estimators = [('lr', lr_clf), ('xgb', xgb_clf), ('rf', rf_clf)], 
                               voting = 'soft', 
                               weights = [1, 1.33, 1])

voting_clf.fit(X_train,y_train)
y_score = voting_clf.predict_proba(X_test)[:,1]
y_pred = voting_clf.predict(X_test)
get_predictions(y_true = y_test, y_pred = y_pred)

# %%
# ## Plot scores, ROC, Precision, Recall, F1 curve, Precision-Recall curve
C.plot_scores(y_true = y_test, y_score = y_score)
fpr_voting, tpr_voting, roc_auc_voting = C.plot_ROC(y_true = y_test, y_score = y_score)
precision_voting, recall_voting, threshold_voting, f1_scores_voting = C.plot_precision_recall_vs_threshold(y_true = y_test, y_score = y_score)
_, _, pr_auc_voting = C.plot_precision_recall(y_true = y_test, y_score = y_score)


# %%
def roc_curve_for_all_models():
    plt.figure(figsize = (16, 12))
    plt.plot(fpr_lr, tpr_lr, label = 'LogisticRegression: AUC = {:.4f}'.format(roc_auc_lr), linewidth = 2)
    plt.plot(fpr_xgb, tpr_xgb, label = 'XGBoost: AUC = {:.4f}'.format(roc_auc_xgb), linewidth = 2)
    plt.plot(fpr_rf, tpr_rf, label = 'Random Forest: AUC = {:.4f}'.format(roc_auc_rf), linewidth = 2)
    plt.plot(fpr_voting, tpr_voting, label = 'VotingClassifier: AUC = {:.4f}'.format(roc_auc_voting), linewidth = 2)
    plt.plot([0, 1], [0, 1], 'k--', linewidth = 2)
    plt.xlim([-0.05, 1])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate(FPR)')
    plt.ylabel('True Positive Rate(TPR)')
    plt.title('Receiver operating characteristic curve (ROC)')
    plt.legend(loc = 'lower right')
    plt.savefig('roc.png')
    plt.show()

# %%
roc_curve_for_all_models()

# %%
def precision_recall_for_all_models () :
    plt.figure(figsize = (16, 12))
    f_scores = np.linspace(0.2, 0.8, num = 4)
    for f_score in f_scores:
        x = np.linspace(0.001, 1)
        y = f_score * x / (2 * x - f_score)
        plt.plot(x[y >= 0], y[y >= 0], color = 'gray', alpha = 0.2)
        plt.annotate('F1 = {:.2f}'.format(f_score), xy = (0.92, y[45] + 0.02))

    plt.plot(recall_lr, precision_lr, label = 'LogisticRegression: AUC = {:.4f}'.format(pr_auc_lr), linewidth = 2)
    plt.plot(recall_xgb, precision_xgb, label = 'XGBoost: AUC = {:.4f}'.format(pr_auc_xgb), linewidth = 2)
    plt.plot(recall_rf, precision_rf, label = 'Random Forest: AUC = {:.4f}'.format(pr_auc_rf), linewidth = 2)
    plt.plot(recall_voting, precision_voting, label = 'VotingClassifier: AUC = {:.4f}'.format(pr_auc_voting), linewidth = 2)
    plt.xlim([0.0, 1])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision - Recall Curve')
    plt.legend(loc = 'lower left')
    plt.savefig('precision_recall.png')
    plt.show()

# %%
precision_recall_for_all_models() 




# %% [markdown]
# ## Select threshold by stratifiefKfold cross-validation
# %%
from selectThreshold import selectThresholdByCV
skf = StratifiedKFold(n_splits = 5)
thresholds = np.array([0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
# classifier = RandomForestClassifier(n_jobs = -1, random_state = 0)
# classifier = xgb.XGBClassifier(n_jobs = -1, n_estimators = 200)
classifier = VotingClassifier (estimators = [('lr', lr_clf), ('xgb', xgb_clf), ('rf', rf_clf)], 
                               voting = 'soft', 
                               weights = [1, 2, 2])


# %%
best_threshold, results = selectThresholdByCV(clf = classifier, X = X_train, y = y_train, 
                                              thresholds = thresholds, cv = skf, 
                                              plot_result = True)
best_threshold
# %%

# %%









