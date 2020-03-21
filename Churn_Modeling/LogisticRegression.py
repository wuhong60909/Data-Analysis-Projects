# # Churn_Modelling
# #### Reference: https://www.kaggle.com/aakash50897/churn-modellingcsv

# Importing Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing dataset
dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values


# ## Data Preprocessing
# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])
labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])
onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X).toarray()
X = X[:, 1:] # Avoiding dummy variable trap! 

# Baseline: France; 
# X: Germany, Spain, CreditScore, Gender(male: 1, Female: 0), Age, Tenure, Balance, NumberOfProducts, HasCrCard, IsActiveMember, EstimatedSalary
# constant: O
# X: x1: O, x2: X, x3: O, x4: O, x5: O, x6: O, x7: O, x8: O, x9: X, x10: O, x11: X.


# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)




# ## Fitting the classifier to the Training set: Logistic Regression
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(solver = 'liblinear', random_state = 0)
classifier.fit(X_train, y_train)
classifier.intercept_
classifier.coef_

# Model Interpretation
import statsmodels.api as sm
X_const = sm.add_constant(X_train)
model = sm.Logit(endog = y_train, exog = X_const).fit()
model.summary()
model.params



'''
import statsmodels.formula.api as smf
model = smf.Logit(endog = y_train, exog = X_train).fit()
model.summary()
'''

# ## Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
def Classification(clf, X, y):
    X_set, y_set = X, y
    y_hat = clf.predict(X_set)
    y_hat = np.reshape(y_hat, -1)
    y_hat[y_hat >= 0.5] = 1
    y_hat[y_hat < 0.5] = 0
    cm = confusion_matrix(y_set, y_hat)
    accuracy = (cm[0, 0] + cm[1, 1])/cm.sum()
    TPR = cm[0, 0]/cm[:, 0].sum() # Sensitivitive, Recall
    TNR = cm[1, 1]/cm[:, 1].sum() # Specificitive
    PPV = cm[0, 0]/cm[0, :].sum() # Positive Predictive Value, Precision
    NPV = cm[1, 1]/cm[1, :].sum() # Negative Predictive Value,  
    F1_score = 2/(1/PPV + 1/TPR)
    summary = {'Accuracy': accuracy, 
               'Positive_Predictive_Value': PPV, 
               'Negative_Predictive_Value': NPV,            
               'Sensitivitive': TPR, 
               'Specificitive': TNR,            
               'F1_score': F1_score, 
               'CM': cm}
    return summary


# ### Training Set
Classification(clf = classifier, X = X_train, y = y_train)

# ### Test Set
Classification(clf = classifier, X = X_test, y = y_test)

# ## Using k-fold Cross Validation to evaluate the model's performance
# Applying k-fold Cross Validation
from sklearn.model_selection import cross_val_score
def cv_score(clf, X, y, cv):
    accuracies = cross_val_score(estimator = clf, X = X, y = y, cv = cv)
    plt.plot(accuracies, '-o')
    plt.axhline(accuracies.mean(), color = 'black', ls = '-')
    plt.axhline(accuracies.mean() + 2 * accuracies.std(), color = 'black', ls = '--')
    plt.axhline(accuracies.mean() - 2 * accuracies.std(), color = 'black', ls = '--')
    plt.xlabel('CV')
    plt.ylabel('Accuracy')
    plt.show()
    return [accuracies.mean(), accuracies.std()]


from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(solver = 'lbfgs', random_state = 0)
Logistic_score = cv_score(clf = classifier, X = X_train, y = y_train, cv = 10)
classifier.fit(X_train, y_train)
Logistic_score.append(classifier.score(X_test, y_test))
print('Logistic Regression: CV score = %0.3f (+/- %0.3f); Test set accuracy = %0.3f' % (Logistic_score[0], 2 * Logistic_score[1], Logistic_score[2]))


# ## Model Selection
# ### Tuning the hyper-parameters to raise the performance: L1, L2.
# Applying Grid Search to find the best model and the best parameters
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
classifier2 = LogisticRegression(solver = 'liblinear', random_state = 0)
parameter = [{'penalty': ['l1'], 'C': np.arange(0.0001, 1, 0.0001)}, 
             {'penalty': ['l2'], 'C': np.arange(0.0001, 1, 0.0001)}]
grid_search = GridSearchCV(estimator = classifier2, 
                           param_grid = parameter,
                           scoring = 'accuracy',
                           cv = 10,
                           n_jobs = -1)
grid_search = grid_search.fit(X_train, y_train)
best_accuracy = grid_search.best_score_
best_parameters = grid_search.best_params_
results = grid_search.cv_results_
best_parameters


# ## Fitting the classifier to the Training set with best_parameters
# Applying k-fold Cross Validation
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
classifier2 = LogisticRegression(C = best_parameters['C'], 
                                 penalty = best_parameters['penalty'], 
                                 solver = 'liblinear', 
                                 random_state = 0)
Logistic_score2 = cv_score(clf = classifier2, X = X_train, y = y_train, cv = 10)
classifier2.fit(X_train, y_train)
Logistic_score2.append(classifier2.score(X_test, y_test))
print('Logistic Regression(L1): CV score = %0.3f (+/- %0.3f); Test set accuracy = %0.3f' % (Logistic_score2[0], 2 * Logistic_score2[1], Logistic_score2[2]))

best_coef = classifier2.coef_
best_L1_norm = np.abs(best_coef).sum()


# ### Training Set
Classification(clf = classifier, X = X_train, y = y_train)


# ### Test Set
Classification(clf = classifier, X = X_test, y = y_test)

# Plot L1 coefficients
from sklearn.linear_model import LogisticRegression
p = len(X_train[0])
c = np.arange(0.0001, 1, 0.001)
Cost = np.zeros(shape = (len(c), p))
coef = np.zeros(shape = (len(c), p))
L1_norm = np.zeros(shape = (len(c), p))
for ii in range(len(c)):
    classifier = LogisticRegression(penalty = 'l1', 
                                    C = c[ii], 
                                    solver = 'liblinear', 
                                    random_state = 0)
    classifier.fit(X_train, y_train)
    coef[ii, :] = classifier.coef_
    L1_norm[ii, :] = np.full((1, p), np.abs(classifier.coef_).sum()) 
    Cost[ii, :] = np.full((1, p), c[ii])

for i in range(p):
    plt.plot(L1_norm[:, i], coef[:, i], '-o', label = i)
plt.axhline(0, color = 'black')
plt.axvline(best_L1_norm, ls = '--', color = 'black')
plt.xlabel('L1_norm')
plt.ylabel('Coefficients')
plt.legend(loc = 'upper left')
plt.show()


# # Conclusion
# 此二元分類問題採用的是 Logistic Regression 作為分類器，並以 k-fold Cross Validation 來評估模型的表現。以預設參數建模得到的 CV score 為 0.808，Test Set 的預測準確度為0.811，經過超參數調整後，新模型的CV score與預測的準確率有些微提升。

print('Logistic Regression: CV score = %0.3f (+/- %0.3f); Test set accuracy = %0.3f' % (Logistic_score[0], 2 * Logistic_score[1], Logistic_score[2]))
print('Logistic Regression(L1): CV score = %0.3f (+/- %0.3f); Test set accuracy = %0.3f' % (Logistic_score2[0], 2 * Logistic_score2[1], Logistic_score2[2]))

