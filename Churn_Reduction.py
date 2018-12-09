
# coding: utf-8

# # Churn Reduction

# In[ ]:


## Imports.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder, StandardScaler, Normalizer
from sklearn.model_selection import StratifiedKFold
#import statsmodels.api as sm
#from statsmodels.stats.outliers_influence import variance_inflation_factor

get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


## Reading the data.

churn_data = pd.read_csv('Train_data.csv')
churn_data.head()


# ### Change of column names
# For the ease of use during analysis

# In[ ]:


churn_data.columns


# In[ ]:


churn_data.columns = ['State', 'Account_length', 'Area_code', 'Phone_number', 'Intl_plan', 'Voicemail_plan',
                      'Number_vmail_message', 'Day_mins', 'Day_calls', 'Day_charges', 'Eve_mins',
                      'Eve_calls', 'Eve_charges', 'Night_mins', 'Night_calls', 'Night_charges', 'Intl_mins',
                      'Intl_calls', 'Intl_charges', 'Cust_serv_calls', 'Churn']


# In[ ]:


## These columns we of binary nature i.e either of (yes/no) type or (True/False) type.
## These have been changed to 1 (for yes and True) and 2 (for no and False).

churn_data.Intl_plan = churn_data.Intl_plan.replace(churn_data.Intl_plan.unique(), (0, 1))
churn_data.Voicemail_plan = churn_data.Voicemail_plan.replace(churn_data.Voicemail_plan.unique(), (1, 0))
churn_data.Churn = churn_data.Churn.replace(churn_data.Churn.unique() , (0, 1))

## Transforming the Labels of states as integers
churn_data.State = LabelEncoder().fit_transform(churn_data.State)


# In[ ]:


## The columns that are categorical in nature are changed to such.

churn_data.State = churn_data.State.astype('category')
churn_data.Area_code = churn_data.Area_code.astype('category')
churn_data.Intl_plan = churn_data.Intl_plan.astype('category')
churn_data.Voicemail_plan = churn_data.Voicemail_plan.astype('category')
churn_data.Churn = churn_data.Churn.astype('category')


# In[ ]:


churn_data.head()


# In[ ]:


churn_data.info()


# In[ ]:


churn_test = pd.read_csv('./Test_data.csv')
churn_test.columns = ['State', 'Account_length', 'Area_code', 'Phone_number', 'Intl_plan', 'Voicemail_plan',
                      'Number_vmail_message', 'Day_mins', 'Day_calls', 'Day_charges', 'Eve_mins',
                      'Eve_calls', 'Eve_charges', 'Night_mins', 'Night_calls', 'Night_charges', 'Intl_mins',
                      'Intl_calls', 'Intl_charges', 'Cust_serv_calls', 'Churn']


# In[ ]:


## Converting the categorical features into levels
churn_test.Intl_plan = churn_test.Intl_plan.replace(churn_test.Intl_plan.unique(), (0, 1))
churn_test.Voicemail_plan = churn_test.Voicemail_plan.replace(churn_test.Voicemail_plan.unique(), (1, 0))
churn_test.Churn = churn_test.Churn.replace(churn_test.Churn.unique(), (0, 1))

churn_test.State = LabelEncoder().fit_transform(churn_test.State)


# In[ ]:


churn_test.State = churn_test.State.astype('category')
churn_test.Area_code = churn_test.Area_code.astype('category')
churn_test.Intl_plan = churn_test.Intl_plan.astype('category')
churn_test.Voicemail_plan = churn_test.Voicemail_plan.astype('category')
churn_test.Churn = churn_test.Churn.astype('category')


# In[ ]:


churn_test.info()


# In[ ]:


churn_test.head()


# ### No missing Value
# From the above table, we infer that in the given data has no missing values.
#
#
# ### Outlier Analysis

# In[ ]:


def fivePtAnalysis(x, y):
    ## Plotting a boxplot for each feature.
    plt.figure()
    print("\n", y)
    plt.boxplot(x)
    plt.show()

    ## Calculating the maximum and minimum for a feature.
    q1, q3 = np.percentile(x, [25, 75])
    iqr = q3 - q1
    maxm, minm = q3 + (1.5 * iqr), q1 - (1.5 * iqr)
    return (maxm, minm)


# In[ ]:


def neutralizeOutliers(x, y, max_val, min_val):
    ## Finding the indices of the outliers in a feature.
    max_in = churn_data[x > max_val].index
    min_in = churn_data[x < min_val].index
    print("\tOutliers above maximum : ", len(max_in))
    print("\tOutliers below minimum : ", len(min_in))

    ## Rounding the maximum and minimum to ceiling.
    if x.dtype == 'int64':
        max_val, min_val = round(max_val), round(min_val)
    col_type = x.dtype

    ## The ouliers lying beyond the maximum limit are reduced to the maximum.
    if len(max_in) > 0:
        for i in max_in:
            churn_data[y].iloc[i] = max_val

    ## The outliers lying beyond the minimum limit are increased to the minimum.
    if len(min_in) > 0:
        for i in min_in:
            churn_data[y].iloc[i] = min_val
    return col_type


# In[ ]:


for i in churn_data:
    ## The outlier analysis is done for all numerical features
    if churn_data[i].dtype == 'float64' or churn_data[i].dtype == 'int64':
        max_val, min_val = fivePtAnalysis(churn_data[i], i)
        col_type = neutralizeOutliers(churn_data[i], i, max_val, min_val)
        if col_type == 'int64':
            churn_data[i] = churn_data[i].astype('int64')


# In[ ]:


churn_data.info()


# In[ ]:


## Style of graphs
sns.set(style="white")
sns.set(style="whitegrid", color_codes=True)


# In[ ]:


churn_data.Churn.value_counts()


# In[ ]:


## Distribution of classes.
sns.countplot(x='Churn', data=churn_data, palette='hls')
plt.show()


# In[ ]:


churn_data.groupby('Churn').mean()


# In[ ]:


## Distribution of classes for each state.
churn_data.groupby(['State', 'Churn']).mean()


# In[ ]:


## Distribution of classes for each area code.
churn_data.groupby(['Area_code','Churn']).mean()


# In[ ]:


pd.crosstab(churn_data.State, churn_data.Churn).plot(kind="bar", figsize=(20,20))
plt.show()


# In[ ]:


## Distribution of classes based on precription of International Plan.
churn_data.groupby(['Intl_plan', 'Churn']).mean()


# In[ ]:


table = pd.crosstab(churn_data.Intl_plan, churn_data.Churn)
table.div(table.sum(1).astype(float), axis=0).plot(kind='bar', stacked=True, figsize=(10,10));


# In[ ]:


## Distribution of classes based on precription of International Plan.
churn_data.groupby(['Voicemail_plan','Churn']).mean()


# In[ ]:


table = pd.crosstab(churn_data.Voicemail_plan, churn_data.Churn)
table.div(table.sum(1).astype(float), axis=0).plot(kind='bar', stacked=True, figsize=(10,10));


# In[ ]:


## Separating the target variable.
target_churn = churn_data['Churn']
target_test_churn = churn_test['Churn']


# In[ ]:


## Feature set.
feat_churn = churn_data.drop(['Churn'], axis =1 )
feat_test_churn = churn_test.drop(['Churn'], axis =1)


# In[ ]:


## Calculating charges per minute.
churn_data['Day_charge_min'] = churn_data['Day_charges'] / churn_data['Day_mins']
churn_test['Day_charge_min'] = churn_test['Day_charges'] / churn_test['Day_mins']

churn_data['Eve_charge_min'] = churn_data['Eve_charges'] / churn_data['Eve_mins']
churn_test['Eve_charge_min'] = churn_test['Eve_charges'] / churn_test['Eve_mins']

churn_data['Night_charge_min'] = churn_data['Night_charges'] / churn_data['Night_mins']
churn_test['Night_charge_min'] = churn_test['Night_charges'] / churn_test['Night_mins']

churn_data['Intl_charge_min'] = churn_data['Intl_charges'] / churn_data['Intl_mins']
churn_test['Intl_charge_min'] = churn_test['Intl_charges'] / churn_test['Intl_mins']


# In[ ]:


## Dropping some features due to two reasons.
## Phone number is dropped because it is like a id.
## Rest are dropped because the features or their derivatives are used.
churn_data = churn_data.drop(['Phone_number', 'Area_code', 'Day_charges', 'Eve_charges',
                              'Night_charges', 'Intl_charges'], axis = 1)
churn_test = churn_test.drop(['Phone_number', 'Area_code', 'Day_charges', 'Eve_charges',
                              'Night_charges', 'Intl_charges'], axis = 1)


# In[ ]:


## Another feature set  with charges per minutes used
feat_churn_1 = churn_data.drop(['Churn'], axis = 1)
feat_test_churn_1 = churn_test.drop(['Churn'], axis = 1)


# In[ ]:


## Phone number and area code are removed from first feature set.
feat_churn =feat_churn.drop(['Phone_number', 'Area_code'], axis =1)
feat_test_churn =feat_test_churn.drop(['Phone_number', 'Area_code'], axis =1)


# In[ ]:


## Feature Scaling.
X = feat_churn.as_matrix().astype(np.float)
y = target_churn.as_matrix().astype(np.float)
X = StandardScaler().fit_transform(X)


# In[ ]:


## Stratified sampling.
skfold = StratifiedKFold(n_splits=4)


# In[ ]:


def confusionMatrix(X, y, model, model_name):
    y_pred = y.copy()
    for train, test in skfold.split(X,y):
        X_tr, X_te = X[train,:], X[test,:]
        y_tr = y[train]
        clf = model()
        clf.fit(X_tr, y_tr)
        y_pred[test] = clf.predict(X_te)
    print(model_name,'\nAccuracy Score\t{:.3f}'.format(accuracy_score(y, y_pred)))
    print('\nConfusion Matrix', '\n', confusion_matrix(y, y_pred))
    plotConfusionHeatMap(confusion_matrix(y, y_pred), model_name)
    print('\nClassification Report\n{}'.format(classification_report(y, y_pred)), '\n\n')


# In[ ]:


def plotConfusionHeatMap(conf_matrix, model_name):
    plt.figure(figsize=(10,10))
    plt.title(model_name)
    sns.heatmap(conf_matrix, annot=True, fmt='')
    plt.show()


# In[ ]:


## Finding accuracy of different algorithms using stratified sampling method.
confusionMatrix(X, y, SVC, 'SUPPORT VECTOR CLASSIFIER')
confusionMatrix(X, y, KNeighborsClassifier, 'K NEAREST NEIGHBOURS CLASSIFICATION')
confusionMatrix(X, y, RandomForestClassifier, 'RANDOM FOREST CLASSIFICATION')
confusionMatrix(X, y, LogisticRegression,'LOGISTIC REGRESSION')
confusionMatrix(X, y, GaussianNB, 'GAUSSIAN NAIVE BAYES CLASSIFICATION')
confusionMatrix(X, y, DecisionTreeClassifier, 'DECISION TREE CLASSIFICATION')


# In[ ]:


## Feature scaling for second feature set.
X_1 = feat_churn_1.as_matrix().astype(np.float)
X_1 = StandardScaler().fit_transform(X_1)


# In[ ]:


## Finding accuracy of different algorithms using stratified sampling method.
confusionMatrix(X_1, y, SVC, 'SUPPORT VECTOR CLASSIFIER')
confusionMatrix(X_1, y, KNeighborsClassifier, 'K NEAREST NEIGHBOURS CLASSIFICATION')
confusionMatrix(X_1,y,RandomForestClassifier, 'RANDOM FOREST CLASSIFICATION')
confusionMatrix(X_1, y, LogisticRegression,'LOGISTIC REGRESSION')
confusionMatrix(X_1, y, GaussianNB, 'GAUSSIAN NAIVE BAYES CLASSIFICATION')
confusionMatrix(X_1, y, DecisionTreeClassifier, 'DECISION TREE CLASSIFICATION')


# In[ ]:


## Choosing Random Forest Classification as it has highest accuracy of all classification.
## 500 decision treeare trained.
RF_model = RandomForestClassifier(n_estimators=500, criterion='entropy', max_depth=10, warm_start=True)
RF_model.fit(X, y)

a = RF_model.feature_importances_
a = 100* (a/a.max())
sorted_idx = np.argsort(a)
pos = np.arange(sorted_idx.shape[0]) + .5

plt.figure(figsize=(10, 10))
plt.barh(pos, a[sorted_idx], align='center', color='#9A5AA2')
plt.yticks(pos, np.asanyarray(feat_churn.columns.tolist())[sorted_idx])
plt.xlabel('Relative Importance')
plt.title('Variable Importance')
plt.show()

## Testing the test dataset.
XX = feat_test_churn.as_matrix().astype(np.float)
XX = StandardScaler().fit_transform(XX)
a = RF_model.predict(XX)
print('The mean accuracy : {}\n'.format(RF_model.score(XX, target_test_churn.as_matrix().astype(np.float)) * 100))


# In[ ]:


RF_model = RandomForestClassifier(n_estimators=400, criterion='entropy', max_depth=10, warm_start=True)
RF_model.fit(X, y)

a = RF_model.feature_importances_
a = 100* (a/a.max())
sorted_idx = np.argsort(a)
pos = np.arange(sorted_idx.shape[0]) + .5

plt.figure(figsize=(10, 10))
plt.barh(pos, a[sorted_idx], align='center', color='#9A5AA2')
plt.yticks(pos, np.asanyarray(feat_churn.columns.tolist())[sorted_idx])
plt.xlabel('Relative Importance')
plt.title('Variable Importance')
plt.show()

XX = feat_test_churn.as_matrix().astype(np.float)
XX = StandardScaler().fit_transform(XX)
a = RF_model.predict(XX)
print('The mean accuracy : {}\n'.format(RF_model.score(XX, target_test_churn.as_matrix().astype(np.float)) * 100))


# In[ ]:


RF_model = RandomForestClassifier(n_estimators=200, criterion='entropy', max_depth=10, warm_start=True)
RF_model.fit(X, y)

a = RF_model.feature_importances_
a = 100* (a/a.max())
sorted_idx = np.argsort(a)
pos = np.arange(sorted_idx.shape[0]) + .5

plt.figure(figsize=(10, 10))
plt.barh(pos, a[sorted_idx], align='center', color='#9A5AA2')
plt.yticks(pos, np.asanyarray(feat_churn.columns.tolist())[sorted_idx])
plt.xlabel('Relative Importance')
plt.title('Variable Importance')
plt.show()

XX = feat_test_churn.as_matrix().astype(np.float)
XX = StandardScaler().fit_transform(XX)
a = RF_model.predict(XX)
print('The mean accuracy : {}\n'.format(RF_model.score(XX, target_test_churn.as_matrix().astype(np.float)) * 100))

