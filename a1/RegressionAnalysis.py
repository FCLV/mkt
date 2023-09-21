#!/usr/bin/env python
# coding: utf-8




import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

np.random.seed(36)


# # 1. Loading Data




df = pd.read_csv('./Customer-Value-Analysis.csv')





df.shape





df.head()





df['Engaged'] = df['Response'].apply(lambda x: 0 if x == 'No' else 1)





df.head()


# # 2. Data Analysis




list(df.columns)


# #### - Engagement Rate




engagement_rate_df = pd.DataFrame(
    df.groupby('Engaged').count()['Response'] / df.shape[0] * 100.0
)





engagement_rate_df





engagement_rate_df.T


# #### - By Renew Offer Type




engagement_by_offer_type_df = pd.pivot_table(
    df, values='Response', index='Renew Offer Type', columns='Engaged', aggfunc=len
).fillna(0.0)

engagement_by_offer_type_df.columns = ['Not Engaged', 'Engaged']





engagement_by_offer_type_df





engagement_by_offer_type_df.plot(
    kind='pie',
    figsize=(15, 7),
    startangle=90,
    subplots=True,
    autopct=lambda x: '%0.1f%%' % x
)

plt.show()


# #### - By Sales Channel




engagement_by_sales_channel_df = pd.pivot_table(
    df, values='Response', index='Sales Channel', columns='Engaged', aggfunc=len
).fillna(0.0)

engagement_by_sales_channel_df.columns = ['Not Engaged', 'Engaged']





engagement_by_sales_channel_df





engagement_by_sales_channel_df.plot(
    kind='pie',
    figsize=(15, 7),
    startangle=90,
    subplots=True,
    autopct=lambda x: '%0.1f%%' % x
)

plt.show()


# #### - Total Claim Amount Distributions




ax = df[['Engaged', 'Total Claim Amount']].boxplot(
    by='Engaged',
    showfliers=False,
    figsize=(7,5)
)

ax.set_xlabel('Engaged')
ax.set_ylabel('Total Claim Amount')
ax.set_title('Total Claim Amount Distributions by Enagements')

plt.suptitle("")
plt.show()





ax = df[['Engaged', 'Total Claim Amount']].boxplot(
    by='Engaged',
    showfliers=True,
    figsize=(7,5)
)

ax.set_xlabel('Engaged')
ax.set_ylabel('Total Claim Amount')
ax.set_title('Total Claim Amount Distributions by Enagements')

plt.suptitle("")
plt.show()


# #### - Income Distributions




ax = df[['Engaged', 'Income']].boxplot(
    by='Engaged',
    showfliers=True,
    figsize=(7,5)
)

ax.set_xlabel('Engaged')
ax.set_xlabel('Income')
ax.set_title('Income Distributions by Enagements')

plt.suptitle("")
plt.show()





df.groupby('Engaged').describe()['Income'].T


# # 3. Dealing with Categorical Variables




df.describe()


# #### - Different ways to handle categorical variables

# ###### 1. factorize




labels, levels = df['Education'].factorize()





labels





levels


# ###### 2. pandas' Categorical variable series




categories = pd.Categorical(
    df['Education'], 
    categories=['High School or Below', 'Bachelor', 'College', 'Master', 'Doctor']
)




categories.categories





categories.codes


# ###### 3. dummy variables




pd.get_dummies(df['Education']).head(10)


# #### - Adding Gender




gender_values, gender_labels = df['Gender'].factorize()
df['GenderFactorized'] = gender_values





gender_values





gender_labels





df


# #### - Adding Education Level




categories = pd.Categorical(
    df['Education'], 
    categories=['High School or Below', 'Bachelor', 'College', 'Master', 'Doctor']
)





categories.codes





categories.categories





df['EducationFactorized'] = categories.codes





df.head()


# # 4. Regression Analysis with Both Continuous and Categorical Variables




y = df['Engaged'] 
X = df[['Customer Lifetime Value',
        'Income',
        'Monthly Premium Auto',
        'Months Since Last Claim',
        'Months Since Policy Inception',
        'Number of Open Complaints',
        'Number of Policies',
        'Total Claim Amount',
        'GenderFactorized',
        'EducationFactorized'
    ]]





from sklearn import model_selection
x_tran,x_test,y_tran,y_test=model_selection.train_test_split(X,y,test_size=0.1)
print(x_test.shape)  # traning and test set





from sklearn.linear_model import LogisticRegression
lr=LogisticRegression(multi_class='ovr',solver='lbfgs',class_weight='balanced')
lr.fit(x_tran,y_tran)
score=lr.score(x_tran,y_tran)
print(score) ## best is 1





from sklearn.metrics import accuracy_score
train_score=accuracy_score(y_tran,lr.predict(x_tran))
test_score=lr.score(x_test,y_test)
print('training set acurrcy rate：',train_score)
print('test set acurracy rate：',test_score)





from sklearn.metrics import recall_score
train_recall=recall_score(y_tran,lr.predict(x_tran),average='macro')
test_recall=recall_score(y_test,lr.predict(x_test),average='macro')
print('training set recall rate：',train_recall)
print('test set recall rate：',test_recall)






y_pro=lr.predict_proba(x_test) ## predict probability
y_prd2 = [list(p>=0.25).index(1) for i,p in enumerate(y_pro)]   ##the threshold is 0.25
train_score=accuracy_score(y_test,y_prd2)
print(train_score)





import statsmodels.api as sm

logit = sm.Logit(y_tran, x_tran) 





logit_fit = logit.fit()





logit_fit.summary()

