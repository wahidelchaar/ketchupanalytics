#!/usr/bin/env python
# coding: utf-8

# In[1]:

import numpy as np
import pandas as pd
pd.set_option('display.expand_frame_repr', False)
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")


# In[2]:

data = pd.read_sas('heinzhunts.sas7bdat')
data.head()


# In[3]:

data = data.sample(frac=1).reset_index().iloc[:,1:]
data.head()


# In[4]:

data.describe()


# In[5]:

data.info()


# In[6]:

for feature in ['HEINZ','HUNTS','FeatHeinz','FeatHunts','DisplHeinz','DisplHunts']:
    data[feature] = data[feature].astype(int)


# In[7]:

data.info()


# In[8]:

plt.figure(figsize=(4,5))
sns.countplot(x='HEINZ',data=data)
plt.grid(axis='y')
plt.xlabel('Brand Choice (1 for Heinz, 0 for Hunts)')
plt.ylabel('Number of Instances')
plt.title('Class Distribution of Outcome Variable')


# In[9]:

sns.kdeplot(data['PRICEHEINZ'], label='Heinz')
sns.kdeplot(data['PRICEHUNTS'], label='Hunts')


# In[10]:

data['priceratio'] = data['PRICEHEINZ']/data['PRICEHUNTS']


# In[11]:

sns.distplot(data['priceratio'])


# In[12]:

# Notice the outliers and the skewness, apply log transformation to make more symmetric.
sns.distplot(np.log(data['priceratio']))


# In[13]:

data['logpriceratio'] = np.log(data['priceratio'])
data.drop(labels=['priceratio','HUNTS'], axis=1, inplace=True)

# Notice that the data centers around zero, indicating possibly no difference in prices, let's test it out.
from scipy.stats import levene, ttest_ind

def ttest_for_difference_means(sample1,sample2,alpha):
    
    # First, test the samples for equal or unequal variances using Levene's test:
    print('\nTesting for difference in sample variances...\n')
    b,p1 = levene(heinz_prices, hunts_prices, center='mean')
    
    # Run independent two-sample test depending on the variances being equal/unequal
    if p1 <= alpha:
        print('Reject null hypothesis that variances are equal. ', 'p-value = {}\n'.format(p1))
        print('Testing for difference in means assuming unequal variances...\n')
        v,p2 = ttest_ind(sample1,sample2,equal_var=False)
        if p2 <= alpha:
            print('Reject the null hypothesis that the means are the same. ', 'p-value = {}\n'.format(p2))
        else:
            print('Do not reject the null hypothesis that the means are the same. ', 'p-value = {}\n'.format(p2))
    
    else:
        print('Do not reject null hypothesis that variances are equal. ', 'p-value = {}'.format(p1))
        print('Testing for difference in means assuming equal variances...\n')
        v,p2 = ttest_ind(sample1,sample2,equal_var=True)
        if p2 <= alpha:
            print('Reject the null hypothesis that the means are the same. ', 'p-value = {}\n'.format(p2))
        else:
            print('Do not reject the null hypothesis that the means are the same. ', 'p-value = {}\n'.format(p2))
            
heinz_prices = data['PRICEHEINZ'].values
hunts_prices = data['PRICEHUNTS'].values

ttest_for_difference_means(heinz_prices, hunts_prices, 0.05)


# In[14]:

data['DisplHeinz*FeatHeinz'] = data['DisplHeinz']*data['FeatHeinz']
data['DisplHunts*FeatHunts'] = data['DisplHunts']*data['FeatHunts']
data.head()


# In[15]:

X = data.iloc[:,3:]
y = data['HEINZ']


# In[16]:

def inferential_logit_model(predictors, response):
    
    # import statsmodels API
    import statsmodels.api as sm
    
    # Instantiate and fit logistic model for inference
    LR = sm.Logit(response, sm.add_constant(predictors))
    result = LR.fit()
    
    # Create odds ratio estimates table
    params = np.exp(result.params)
    conf_1 = result.conf_int()[0] # at 2.5%
    conf_2 = result.conf_int()[1] # at 97.5%
    odds = pd.DataFrame(data=[params, conf_1, conf_2]).transpose().drop('const')
    odds.rename(columns={"Unnamed 0":"Odds Ratio Estimates", 0:"2.5%", 1:"97.5%"}, inplace=True)
    
    # Print the maximum likelihood estimates and odds ratio estimates tables with diagnostics
    print(result.summary(),'\n',odds)


# In[17]:

inferential_logit_model(X,y)


# In[18]:

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

X = X.iloc[:,:5]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

logit = LogisticRegression(C=10e9)
logit.fit(X_train, y_train)
y_pred = logit.predict(X_test)

from sklearn.metrics import accuracy_score, recall_score, precision_score, classification_report

print(accuracy_score(y_test, y_pred),recall_score(y_test, y_pred),precision_score(y_test, y_pred))

print(classification_report(y_test, y_pred))


# In[19]:

from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import precision_score

def find_best_params(model,parameters,folds,X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test):
    
    skf = StratifiedKFold(n_splits=folds)

    grid = GridSearchCV(estimator=model, param_grid=parameters, cv=skf,
                        scoring='precision', return_train_score=True)
    grid.fit(X_train,y_train)
    
    print('Paremeter combination that yields best precision: {}\n'.format(grid.best_params_))
    print(pd.DataFrame(grid.cv_results_)[['params','mean_test_score','mean_train_score']])
    
    y_pred = grid.predict(X_test)
    
    print('\nPrecision on test set: {}'.format(round(precision_score(y_test, y_pred),6)))
    

# In[20]:

parameters = {'C':[0.1,1,10,100,1000], 'penalty':['l1','l2']}
LR = LogisticRegression()

find_best_params(model=LR, parameters=parameters, folds=5)


# In[21]:

def false_positives(y_true, y_pred):
    return np.sum((y_true==0) & (y_pred==1))

def false_negatives(y_true, y_pred):
    return np.sum((y_true==1) & (y_pred==0))

def find_threshold_for_minimum_cost(model, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test):

    model.fit(X_train, y_train)
    probs = model.predict_proba(X_test)[:,1]
    
    levels = []
    FP = []
    FN = []

    for threshold in np.linspace(0,1,100):
        levels.append(threshold)
        class_pred = [1 if p>=threshold else 0 for p in probs]
        FP.append(false_positives(y_test,np.array(class_pred)))
        FN.append(false_negatives(y_test,np.array(class_pred)))

    Cost_df = pd.DataFrame(data=[levels,FP,FN]).transpose()
    Cost_df.rename(columns={0:'Threshold', 1:'False Positives', 2:'False Negatives'}, inplace=True)
    Cost_df['Cost'] = Cost_df['False Positives']*1 + Cost_df['False Negatives']*0.25

    plt.figure(figsize=(12,5))
    plt.plot(Cost_df['Threshold'], Cost_df['Cost'], 'r')
    plt.xlabel('Classification Threshold')
    plt.ylabel('Cost')
    plt.title('Cost of Mistargeted Marketing as a function of Classification Threshold')
    plt.grid()
    
    print(Cost_df.sort_values(by='Cost').head(5))


# In[22]:

logit_optimal = LogisticRegression(C=10, penalty='l1')
find_threshold_for_minimum_cost(logit_optimal)

from sklearn.ensemble import BaggingClassifier, GradientBoostingClassifier
from sklearn.model_selection import cross_val_score

bag_logit = BaggingClassifier(base_estimator=logit_optimal)
boost_logit = GradientBoostingClassifier()

bag_logit.fit(X=X_train, y=y_train)
boost_logit.fit(X=X_train, y=y_train)

scores_bag = cross_val_score(estimator=bag_logit, cv=5, scoring='precision', X=X, y=y)

for i in scores_bag:
    print(i)
    
scores_boost = cross_val_score(estimator=boost_logit, cv=5, scoring='precision', X=X, y=y)
 
for i in scores_boost:
    print(i)
                              

find_threshold_for_minimum_cost(bag_logit)
find_threshold_for_minimum_cost(boost_logit)


# In[25]:

from sklearn.ensemble import RandomForestClassifier

rfc = RandomForestClassifier()
parameters = {'n_estimators': [200, 500],
              'max_features': ['auto', 'sqrt', 'log2'],
              'max_depth' : [4,5,6,7,8],
              'criterion' : ['gini', 'entropy']}

find_best_params(model=rfc, parameters=parameters, folds=5)


# In[26]:

rfc_optimal = RandomForestClassifier(criterion='gini', max_depth=8, 
                                     max_features='auto', n_estimators=500, oob_score=True)

find_threshold_for_minimum_cost(model=rfc_optimal)


# In[27]:
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

X_train_scaled, X_test_scaled, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

svc = SVC()
parameters = {'C': [1, 10, 100, 1000], 
              'gamma': [0.1, 1, 10, 100]}

find_best_params(model=svc, parameters=parameters, folds=5, X_train=X_train_scaled, X_test=X_test_scaled)


# In[28]:

svc_optimal = SVC(C=100, gamma=100, probability=True)

find_threshold_for_minimum_cost(model=svc_optimal, X_train=X_train_scaled, X_test=X_test_scaled)


# In[29]

# =============================================================================
# Summary of all models and their performance
# =============================================================================

# logit cost table
logit_optimal = LogisticRegression(C=10, penalty='l1')

logit_optimal.fit(X_train, y_train)
probs = logit_optimal.predict_proba(X_test)[:,1]

levels = []
FP = []
FN = []
cost = []

for threshold in np.linspace(0,1,100):
    levels.append(threshold)
    class_pred = [1 if p>=threshold else 0 for p in probs]
    FP.append(false_positives(y_test,np.array(class_pred)))
    FN.append(false_negatives(y_test,np.array(class_pred)))
    
Cost_df_logit = pd.DataFrame(data=[levels,FP,FN]).transpose()
Cost_df_logit.rename(columns={0:'Threshold', 1:'False Positives', 2:'False Negatives'}, inplace=True)
Cost_df_logit['Cost'] = Cost_df_logit['False Positives']*1 + Cost_df_logit['False Negatives']*0.25

# SVC cost table
svc_optimal = SVC(C=1000, gamma=10, probability=True)

svc_optimal.fit(X_train, y_train)
probs = svc_optimal.predict_proba(X_test)[:,1]

levels = []
FP = []
FN = []
cost = []

for threshold in np.linspace(0,1,100):
    levels.append(threshold)
    class_pred = [1 if p>=threshold else 0 for p in probs]
    FP.append(false_positives(y_test,np.array(class_pred)))
    FN.append(false_negatives(y_test,np.array(class_pred)))
    
Cost_df_svc = pd.DataFrame(data=[levels,FP,FN]).transpose()
Cost_df_svc.rename(columns={0:'Threshold', 1:'False Positives', 2:'False Negatives'}, inplace=True)
Cost_df_svc['Cost'] = Cost_df_svc['False Positives']*1 + Cost_df_svc['False Negatives']*0.25

# RFC cost table
rfc_optimal = RandomForestClassifier(criterion='gini', max_depth=8, 
                                     max_features='auto', n_estimators=500)

rfc_optimal.fit(X_train, y_train)
probs = rfc_optimal.predict_proba(X_test)[:,1]

levels = []
FP = []
FN = []
cost = []

for threshold in np.linspace(0,1,100):
    levels.append(threshold)
    class_pred = [1 if p>=threshold else 0 for p in probs]
    FP.append(false_positives(y_test,np.array(class_pred)))
    FN.append(false_negatives(y_test,np.array(class_pred)))
    
Cost_df_rfc = pd.DataFrame(data=[levels,FP,FN]).transpose()
Cost_df_rfc.rename(columns={0:'Threshold', 1:'False Positives', 2:'False Negatives'}, inplace=True)
Cost_df_rfc['Cost'] = Cost_df_rfc['False Positives']*1 + Cost_df_rfc['False Negatives']*0.25

# Get coordinates for (threshold, minimum_cost) for each model:
min_logit = Cost_df_logit['Cost'].min()
threshold_logit = Cost_df_logit.loc[Cost_df_logit['Cost'] == min_logit]['Threshold'].values.item()
logit_coords = '({},{})'.format(round(threshold_logit,2), min_logit)
min_svc = Cost_df_svc['Cost'].min()
threshold_svc = Cost_df_svc.loc[Cost_df_svc['Cost'] == min_svc]['Threshold'].values.item()
svc_coords = '({},{})'.format(round(threshold_svc,2), min_svc)
min_rfc = Cost_df_rfc['Cost'].min()
threshold_rfc = Cost_df_rfc.loc[Cost_df_rfc['Cost'] == min_rfc]['Threshold'].values.item()
rfc_coords = '({},{})'.format(round(threshold_rfc,2), min_rfc)

# Plot costs of mistargeted marketing for all three models:
plt.figure(figsize=(12,7.4))

plt.plot(levels, Cost_df_logit['Cost'], 'r', label='Logit')
plt.axhline(y=min_logit, ls='--', c='black',lw=1.3)
plt.annotate(logit_coords, xy=(1.05,min_logit), va='center', 
             xytext=(1.06,min_logit), fontsize='medium', 
             fontweight='semibold')

plt.plot(levels, Cost_df_svc['Cost'], 'b', label='SVC')
plt.axhline(y=min_svc, ls='--', c='black',lw=1.3)
plt.annotate(svc_coords, xy=(1.05,min_svc), va='center', 
             xytext=(1.06,min_svc), fontsize='medium', 
             fontweight='semibold')

plt.plot(levels, Cost_df_rfc['Cost'], 'g', label='RandomForest')
plt.axhline(y=min_rfc, ls='--', c='black', lw=1.3)
plt.annotate(rfc_coords, xy=(1.05,min_rfc), va='center', 
             xytext=(1.06,min_rfc), fontsize='medium', 
             fontweight='semibold')

plt.xlabel('Classification Threshold')
plt.ylabel('Cost')
plt.title('Cost of Mistargeted Marketing as a function of Classification Threshold')
plt.legend()


# In[30]





