# Credit_Card_Fraud_Detection

Reference: <https://www.kaggle.com/mlg-ulb/creditcardfraud>

## Description

這是一筆附有盜刷註記的信用卡交易資料，其中盜刷交易占整體資料集為0.17%，為極度不平衡資料。資料裡的特徵變量是連續型變量，其中特徵 V1, V2, …V28 是經過PCA轉換後的主成分 (Principle Components)，而 'Time' 與 'Amount' 是未經PCA轉換的變量。'Time' 是每筆交易與第一筆交易之間所經過的秒數，'Amount' 是每筆交易的金額。'Class' 是反應變量，發生詐欺交易註記為1，正常交易註記為0。

## 分析要點

這份分析案例的命題為盜刷偵測，有標籤註記的資料，希望藉由特徵變量去偵測是否為詐欺交易，可視為二元分類問題。以 F1 score 作為分類成效的評分準則，以下有幾項分析要點:

1. Exploratory Data Analysis (EDA)
2. 特徵選取 (Feature Selection)
3. 定義模型表現
4. 分類演算法
5. 模型優化
6. 比較分類演算法表現準則
7. 分類門檻值

## 1. EDA

第一步，以初步的探索式分析概略觀察整體資料，首先盜刷交易占整體資料集為0.17%，為極度不平衡資料，再來是特徵變量皆為連續型變量，沒有類別型變量，因此不需要做特別資料前處理，這是一個單純以連續型特徵變量做預測的二元分類問題。

## 2. 特徵選取 (Feature Selection)

避免使用過多對分類無益的特徵變量，造成模型過度擬和 (overfitting) 問題，所以建立二元分類問題的模型之前，可先對每一個特徵在不同類別下作檢定是否為不同群體。若在不同類別下的特徵變量有顯著差異，則可以考慮加進分類模型。採用 Two-sample T test/Two-sample K-S test，檢定統計量的數字愈大 (i.e. pvalue 愈小) 則在不同類別下的差異愈顯著，以 pvalue 由小到大排名，以顯著性愈高的特徵變量優先選擇。

```python
from scipy.stats import ttest_ind, ks_2samp
```

## 3. 定義模型表現

- Confusion matrix (混淆矩陣):
  - True Positive (TP): 預測結果為 **陽** 性，實際上是 **陽** 性。
  - True Negative (TN): 預測結果為 **陰** 性，實際上是 **陰** 性。
  - False Positive (FP): 預測結果為 **陽** 性，實際上是 **陰** 性 (偽陽性)。
  - False Negative (FN): 預測結果為 **陰** 性，實際上是 **陽** 性 (偽陰性)。

- Metrics:
  - Accuracy (準確率): ( TP + TN ) / ( TP + TN + FP + FN )
  - Precision (精確率): TP / ( TP + FP )
  - Recall/TPR (召回率/真陽性率): TP / ( TP + FN )
  - FPR (偽陽性率): FP / ( TN + FP )
  - F1 score: 2 x ( Precision x Recall ) / ( Precision + Recall )

- Roc Curve: 在不同的門檻值下得到不同的真陽性率 (TPR，即 Recall) 與偽陽性率 (FPR) 所繪製的曲線，x-軸為 FPR， y-軸為 TPR。

- Precision-Recall Curve: 在不同的門檻值下得到不同的 Precision、Recall 所繪製的曲線，x-軸為 Recall， y-軸為 Precision。

```python
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import classification_report
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.metrics import precision_recall_curve, average_precision_score
```

## 4. 分類演算法

分類演算法的選擇，使用的是 scikit-learn 中的 Logistic Regression & Random Forest classifiers，以及近年來流行的機器學習框架如 XGBoost, LightGBM, CatBoost.

```python
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
```

## 5. 模型優化

在訓練模型之前，會需要先給定模型一些參數值後再做計算。希望能調整這些**超參數**(**hyperparameters**) 來提升模型準度，可採用 GridsearchCV 或是 RandomizedsearchCV 尋找超參數來優化模型。因為資料是極度不平衡資料，使用 StratifiedKFold 分層抽樣確保抽樣時各個類別下資料樣本數維持固定比例。

```python
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, StratifiedKFold
```

## 6. 比較分類演算法表現準則

使用 ROC curve? Precision-Recall curve?

**結論**: 使用 Precision-Recall curve，不要用 ROC curve。

以 Naive Bayes 和 Logistic Regression 為例，以 f1-score 作為評分標準，使用兩個演算法各自訓練好模型後，用測試集驗證分類結果，可以觀察到 Naive Bayes 的 f1-score 只有 0.08，而 Logistic Regression 則有 0.63，在預測的表現上 Logistic Regression 是比 Naive Bayes 明顯要好得多，而在 ROC curve 中卻沒有明顯的差異，這是因為 ROC curve 同時考慮了陽性樣本以及陰性樣本，在類別分布極不平均的情況下，陰性樣本遠多於陽性，使得 ROC curve 的 x-軸 FPR 的增長會被稀釋，而 Precision-Recall curve 只注重在陽性樣本，因此在極度不平衡資料，使用 Precision-Recall curve 會有比較好的鑑別度，ROC curve 較適合用在類別平均的案例。

Naive Bayes

```python
              precision    recall  f1-score   support

           0       1.00      0.98      0.99     85335
           1       0.04      0.81      0.08       108

    accuracy                           0.98     85443
   macro avg       0.52      0.89      0.54     85443
weighted avg       1.00      0.98      0.99     85443
```

Logistic Regression

```python
              precision    recall  f1-score   support

           0       1.00      1.00      1.00     85335
           1       0.80      0.52      0.63       108

    accuracy                           1.00     85443
   macro avg       0.90      0.76      0.81     85443
weighted avg       1.00      1.00      1.00     85443
```

ROC curve

![image1](https://github.com/wuhong60909/Data-Analysis-Projects/blob/master/Credit_Card_Fraud_Detection/Figure/lr_vs_nb_roc.png?raw=true "Logistic Regression vs Naive Bayes ROC curve")

Precision-Recall curve

![image2](https://github.com/wuhong60909/Data-Analysis-Projects/blob/master/Credit_Card_Fraud_Detection/Figure/lr_vs_nb_pr.png?raw=true "Logistic Regression vs Naive Bayes P-R curve")

## 7. 分類門檻值

資料餵入模型得到的輸出值為機率值，分類門檻的預設值為 0.5，輸出值 <= 0.5 分為第 0 類，反之輸出值 > 0.5 則分為第 1 類，藉由調整預設值提升分類準度。
