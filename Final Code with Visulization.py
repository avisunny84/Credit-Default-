# -*- coding: utf-8 -*-
"""
Created on Fri Sep  7 11:34:50 2018

@author: AVINASH CHAHAR
"""

#%%
import pandas as pd 
import numpy as np 
#%%
#Reading the txt file in dataframe
df = pd.read_csv(r'C:\Users\AVINASH CHAHAR\Documents\imarticusR\py\GP\XYZCorp_LendingData.txt',delimiter='\t', header =0) 
#%%
#Setting up the column width to ensure that all columns are displayed
pd.set_option('display.max_columns',None)
#%%
df['emp_length'].values
#%% 
#ing data in new dataframe for operating on data
credit_df = df.copy ()
#%%
count_nul = credit_df.isnull().sum()
#%%
percent_null =round(credit_df.isnull().sum()/len(credit_df) * 100,2)
#%%
miss_val= pd.DataFrame({'counts':count_nul,'%age missing':percent_null})
#%%
miss_val.sort_values(by=['%age missing'],ascending=False)
#%%
credit_df = credit_df.loc[:,credit_df.isnull().sum()/len(credit_df) <.76 ] 
#%%
credit_df.shape
#%%
#  function for replacing and formatting
def EmpLength(year):
    if year == '< 1 year':
        return 0    
    elif year == '10+ years': 
        return 10
    else:
        return float(str(year).rstrip(' years'))
credit_df['emp_length'] = credit_df['emp_length'].map(EmpLength)
##credit_df = credit_df.drop('emp_length',axis =1 )
credit_df['emp_length'].value_counts()
#%%
credit_df.isnull().sum()
#%%
credit_df['emp_length'].fillna(value=0, inplace=True)
#%%
credit_df['mths_since_last_major_derog'].fillna(value=0, inplace=True)
#%%
credit_df['mths_since_last_delinq'].fillna(credit_df['mths_since_last_delinq'].median(), inplace=True)
#%%
# Replacing blank by mmedian bcoz original data of "tot_coll_amt" had 677062 observation contains zero
credit_df['tot_coll_amt'].fillna( credit_df['tot_coll_amt'].median(), inplace=True)
#%%
#replacing blank by mean
for value in ['revol_util','collections_12_mths_ex_med',
              'tot_cur_bal','total_rev_hi_lim']:
    credit_df[value].fillna( credit_df[value].mean(),inplace=True)  
#%% #Replacing null values in emp_title and title with Unknown
#credit_df['emp_title'].fillna('Unknown',inplace = True)
#credit_df['title'].fillna('Unknown',inplace = True)
credit_df.columns
#%%
credit_df['issue_d']=pd.to_datetime(credit_df['issue_d'])
#%%
credit_df['last_credit_pull_d']= pd.to_datetime(credit_df['last_credit_pull_d'].fillna("2016-01-01")).apply(lambda x: int(x.strftime('%m')))
#%%
credit_df['earliest_cr_line']= pd.to_datetime(credit_df['earliest_cr_line'].fillna('2001-08-01')).apply(lambda x: int(x.strftime('%m')))
#%%
credit_df['last_pymnt_d']= pd.to_datetime(credit_df['last_pymnt_d'].fillna('2016-01-01')).apply(lambda x: int(x.strftime('%m')))
#%%
credit_df['next_pymnt_d']= pd.to_datetime(credit_df['next_pymnt_d'].fillna('2016-02-01')).apply(lambda x:int(x.strftime("%Y")))
#%% 
credit_df.isnull().sum()
#%%
df.select_dtypes(include =object).apply(pd.Series.nunique,axis=0)
#%%
#Label Encoding
colname = ['term','grade', 'home_ownership','verification_status','pymnt_plan',
           'initial_list_status','application_type','purpose']
from sklearn import preprocessing
le={}
type(le)
for x in colname:
    le[x]=preprocessing.LabelEncoder()
for x in colname:
    credit_df[x]=le[x].fit_transform(credit_df.__getattr__(x)) 
    
#%% ,'pymnt_plan','total_rec_late_fee'
credit_new_df=pd.DataFrame(credit_df,columns=['loan_amnt','int_rate','term','installment','grade','emp_length','home_ownership','annual_inc',
                                              'verification_status','issue_d','purpose','dti','delinq_2yrs','earliest_cr_line','mths_since_last_delinq',
                                              'open_acc','pub_rec','revol_bal','revol_util','total_acc','initial_list_status','out_prncp',
                                              'total_pymnt','total_rec_int','recoveries','last_pymnt_d','last_pymnt_amnt','next_pymnt_d',
                                              'collections_12_mths_ex_med','mths_since_last_major_derog','application_type','acc_now_delinq','tot_coll_amt',
                                              'tot_cur_bal','last_credit_pull_d','default_ind'])
                                              
#%%
credit_new_df.shape#'funded_amnt','funded_amnt_inv',,'policy_code'
#  annual_inc,open_acc,collections_12_mths_ex_med,mths_since_last_major_derog,application_type
#%%
credit_new_df.term.value_counts()
#%%
credit_new_df.isnull().sum()
#%%
# Data Visulization
import seaborn as sns
import matplotlib.pyplot as plt
#%%
# Visulizating Heatmap
plt.figure(figsize=(30, 30))
corr_df = credit_new_df.corr(method = "pearson")
print(corr_df)

sns.heatmap(corr_df,annot=True,vmax=1.0,vmin=-1.0)
#%%
# loan_amnt column
plt.figure(figsize=(12,6))
plt.subplot(121)
g = sns.distplot(credit_new_df["loan_amnt"])
g.set_xlabel("Loan Amount", fontsize=12)
g.set_ylabel("Frequency Dist", fontsize=12)
g.set_title("Frequency Distribuition", fontsize=20)
plt.show()
#%%
fig = credit_df.groupby(['grade'])['loan_amnt'].sum().plot.bar()
fig.set_ylabel('Loan amount disbursed')
#%%
fig = credit_df.groupby(['grade'])['int_rate'].mean().plot.bar()
fig.set_ylabel('Interest Rate')
#%%
fig = credit_df.groupby(['grade'])["total_pymnt"].mean().plot.bar()
fig.set_ylabel('Interest Rate')
#%%
fig = credit_df.groupby(['grade'])['int_rate'].mean().plot.bar()
fig.set_ylabel('Interest Rate')
#%%
fig = credit_df.groupby(["purpose"])['loan_amnt'].sum().plot.bar()
fig.set_ylabel('Loan amount disbursed')

#%%
credit_df.columns
#%%
# Home Ownership v/s Loan Amount
plt.figure(figsize = (10,6))
g = sns.violinplot(x="grade",y="loan_amnt",data=credit_df,
               kind="violin",
               split=True,palette="hls",
               hue="default_ind")
g.set_title("Grade - Loan Distribuition", fontsize=20)
g.set_xlabel("Homer Ownership", fontsize=15)
g.set_ylabel("Loan Amount", fontsize=15)

#%% 
# Disperssion Plot of Insatllment
sns.distplot(credit_df['installment'])
plt.show()
#%%
# Interest rate with Grade
fig = plt.figure(figsize=(18,8))
sns.violinplot(x="default_ind",y="int_rate",data=df, hue="grade")
plt.title("Grade - Interest Rate", fontsize=20)
plt.xlabel("TARGET", fontsize=15)
plt.ylabel("Interest Rate", fontsize=15)
#%%
# Which state has most defaulted values 
fig = plt.figure(figsize=(18,10))
df[df["default_ind"]==1].groupby('addr_state')["default_ind"].count().sort_values().plot(kind='barh')
plt.ylabel('State',fontsize=15)
plt.xlabel('Number of loans',fontsize=15)
plt.title('Number of defaulted loans per state',fontsize=20)
#%%
# Which state has most non defaulted values 
fig = plt.figure(figsize=(18,10))
df[df["default_ind"]==0].groupby('addr_state')["default_ind"].count().sort_values().plot(kind='barh')
plt.ylabel('State',fontsize=15)
plt.xlabel('Number of loans',fontsize=15)
plt.title('Number of non defaulted loans per state',fontsize=20)

#%%
# Data Visulization of Int_rate
credit_new_df['int_round'] = credit_new_df['int_rate'].round(0).astype(int)
plt.figure(figsize = (10,8))

plt.subplot(211)
g = sns.distplot(np.log(credit_new_df["int_rate"]))
g.set_xlabel("", fontsize=12)
g.set_ylabel("Distribuition", fontsize=12)
g.set_title("Int Rate Log distribuition", fontsize=20)

plt.subplot(212)
g1 = sns.countplot(x="int_round",data=credit_new_df, 
                   palette="Set2")
g1.set_xlabel("Int Rate", fontsize=12)
g1.set_ylabel("Count", fontsize=12)
g1.set_title("Int Rate Normal Distribuition", fontsize=20)
plt.subplots_adjust(wspace = 0.2, hspace = 0.6,top = 0.9)

plt.show()
#%%
#Spliting Data Into Test
test = credit_new_df.loc[credit_new_df['issue_d']>='2015-06-01',:]
#%%
#Spliting Data Into Train
train =credit_new_df.loc[credit_new_df['issue_d']<'2015-06-01',:]

#%%
#DATE issue_d
test['issue_d']= test['issue_d'].apply(lambda x: int(x.strftime('%Y')))
#%%
train['issue_d']= train['issue_d'].apply(lambda x: int(x.strftime('%Y')))
#%%
X_train = train.values[:,:-1]
Y_train = train.values[:,-1]
#%%
X_test = test.values[:,:-1]
Y_test = test.values[:,-1]
#%%
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

scaler.fit(X_train)

X_train = scaler.transform(X_train)

print(X_train)
#%%
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

scaler.fit(X_test)

X_test= scaler.transform(X_test)

print(X_test)
#%%
# Building Model using X_train,Y_train
from sklearn.linear_model import LogisticRegression 
classifier = (LogisticRegression())
classifier.fit(X_train,Y_train)
#%%
# Predicting Model Using X_test
Y_pred=classifier.predict(X_test)
#%%
#Confusion Matrix
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
print("Confusion Matrix :")
cfm=confusion_matrix(Y_test,Y_pred)
print(cfm)

print("Classification Report:")
print(classification_report(Y_test,Y_pred))

acc=accuracy_score(Y_test,Y_pred)
print("Accuracy of the model: ",acc)

#%%
#Tuning the model by adjusting threshold 
y_pred_prob = classifier.predict_proba(X_test)
print(y_pred_prob)
#%% Tunning the model 
#Error at threshold 0.45 : 96  , type 2 error : 65  , type 1 error: 31
#Error at threshold 0.5 : 103  , type 2 error : 64  , type 1 error: 39
#Error at threshold 0.55 : 104  , type 2 error : 63  , type 1 error: 41
#Error at threshold 0.60 : 109  , type 2 error : 63  , type 1 error: 46
for a in np.arange(0,1,0.05):
    predict_mine = np.where(y_pred_prob[:,0] < a,1,0)
    cfm=confusion_matrix(Y_test.tolist(),predict_mine)
    total_err=cfm[0,1]+cfm[1,0]
    print("Error at threshold", a, ":", total_err, " , type 2 error :", \
          cfm[1,0]," , type 1 error:", cfm[0,1])

#%%
#Using kfold_cross_validation
classifier=(LogisticRegression())
from sklearn import cross_validation
kfold_cv=cross_validation.KFold(n=len(X_train),n_folds=10)
print(kfold_cv)
#%%
#running the model using scoring metric as accuracy
kfold_cv_result=cross_validation.cross_val_score(estimator=classifier,X=X_train,
y=Y_train, cv=kfold_cv)
print(kfold_cv_result)
#finding the mean
print(kfold_cv_result.mean())
#%%
for train_value, test_value in kfold_cv:
    classifier.fit(X_train[train_value], Y_train[train_value]).predict(X_train[test_value])
Y_pred=classifier.predict(X_test)   
print(list(zip(Y_test,Y_pred)))
#%%
# Cunfusion Matrix of K-Fold Validation 
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
print("Confusion Matrix:")
cfm=confusion_matrix(Y_test.tolist(),Y_pred)
print(cfm)
print("Classification Report:")
print(classification_report(Y_test.tolist(),Y_pred))
acc=accuracy_score(Y_test,Y_pred)
print("Accuracy of the model: ",acc)

#%%
# AUC  on threshold of 0.5
from sklearn import metrics
#preds = classifer.predict_proba(X_test)[:,0]
fpr,tpr, threshold = metrics.roc_curve(Y_test.tolist(), Y_pred)
auc = metrics.auc(fpr,tpr)
print(auc)
#%%

import matplotlib.pyplot as plt
plt.title('Reciver Opertaing Characteristcs')
plt.plot(fpr, tpr, 'b', label = auc)
plt.legend(loc= 'lower right')
plt.plot([0, 1], [0, 1], 'r--')
plt.xlim([0,1])
plt.ylim([0, 1])
plt.xlabel('False Positive rate')
plt.ylabel('True Positive rate')

plt.show()

#%%
#Running Decision Tree Model
from sklearn.tree import DecisionTreeClassifier

model_DecisionTree = DecisionTreeClassifier()
model_DecisionTree.fit(X_train,Y_train)
#%%
#fit the model on the data and predict the values 

Y_pred = model_DecisionTree.predict(X_test)
#print(Y_pred)
print(list(zip(Y_test,Y_pred)))
#%%
#Confusion matrice
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

cfm=confusion_matrix(Y_test,Y_pred)
print(cfm)

print("Classification Report:")
print(classification_report(Y_test,Y_pred))

acc=accuracy_score(Y_test,Y_pred)
print("Accuracy of the model: ",acc)
#%%
credit_new_df['issue_d']= credit_new_df['issue_d'].apply(lambda x: int(x.strftime('%Y')))
#%%
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
#%%
array = credit_new_df.values
#%%
array
#%%
X = array[:,:-1]
Y = array[:,-1]
#%%
# feature extraction
test = SelectKBest(score_func=chi2, k=4)
fit = test.fit(X, Y)
#%%
import numpy
#%%
# summarize     scores
numpy.set_printoptions(precision=3)
print(fit.scores_)
features = fit.transform(X)
# summarize selected features
print(features[0:5,:])
#%%
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
#%%
# feature extraction
model = LogisticRegression()
#%%
rfe = RFE(model, 3)
fit = rfe.fit(X, Y)
#%%
print("Num Features: %d" % fit.n_features_,)
print("Selected Features: %s" % fit.support_,)
print("Feature Ranking: %s" % fit.ranking_,)
#%%
credit_new_df.columns