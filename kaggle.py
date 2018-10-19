from sklearn.model_selection import cross_val_score
from sklearn import linear_model
from sklearn import metrics
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import os
os.chdir("E:\data")

train = pd.read_csv('train.csv',header = 0,index_col=0)
test  = pd.read_csv('test.csv',header = 0,index_col=0)

sns.distplot(train["SalePrice"])
plt.show()

data = train.corr()
sns.heatmap(data)
plt.show()
# In[*]
data = train.corr()
data["SalePrice"].sort_values()

train = train.drop(['BsmtHalfBath',
                    'BsmtFinSF2',
                    '3SsnPorch',
                    'MoSold',
                    'PoolArea',
                    'ScreenPorch',
                    'BedroomAbvGr'], axis=1)

test = test.drop(['BsmtHalfBath',
                    'BsmtFinSF2',
                    '3SsnPorch',
                    'MoSold',
                    'PoolArea',
                    'ScreenPorch',
                    'BedroomAbvGr'], axis=1)

sns.lmplot(x="OverallQual", y="SalePrice",
           data=train, fit_reg=False, scatter=True)
plt.show()

# In[*]


sns.lmplot(x="TotalBsmtSF", y="SalePrice",
           data=train, fit_reg=False, scatter=True)
plt.show()

for col in train.columns:
    if train[col].isnull().sum() > 0:
        print(col, train[col].isnull().sum())

# In[*]

train = train.drop(["MiscFeature", "PoolQC", "Alley",
                    "Fence", 'FireplaceQu'], axis=1)

test = test.drop(["MiscFeature", "PoolQC", "Alley",
                  "Fence", 'FireplaceQu'], axis=1)

# In[*]
print(train.describe())

# In[*]
all_data = pd.concat((train, test))
# In[*]
for col in train.columns:
    if train[col].isnull().sum() > 0:
        if train[col].dtypes == 'object':
            val = all_data[col].dropna().value_counts().idxmax()
            train[col] = train[col].fillna(val)
        else:
            val = all_data[col].dropna().mean()
            train[col] = train[col].fillna(val)
            # In[*]
for col in test.columns:
    if test[col].isnull().sum() > 0:
        if test[col].dtypes == 'object':
            val = all_data[col].dropna().value_counts().idxmax()
            test[col] = test[col].fillna(val)
        else:
            val = all_data[col].dropna().mean()
            test[col] = test[col].fillna(val)

    # In[*]

for col in all_data.select_dtypes(include=[object]).columns:
    train[col] = train[col].astype('category',
                                   categories=all_data[col].dropna().unique())

    test[col] = test[col].astype('category',
                                 categories=all_data[col].dropna().unique())
    # In[*]
for col in train.columns:
    if train[col].dtype.name == 'category':
        tmp = pd.get_dummies(train[col], prefix=col)
        train = train.join(tmp)
        train = train.drop(col, axis=1)
    # In[*]
for col in test.columns:
    if test[col].dtype.name == 'category':
        tmp = pd.get_dummies(test[col], prefix=col)
        test = test.join(tmp)
        test = test.drop(col, axis=1)

    # In[*]

for col in train.columns:
    if train[col].isnull().sum() > 0:
        print(col, train[col].isnull().sum())

lr = linear_model.LinearRegression()
X = train.drop("SalePrice", axis=1)
y = np.log(train["SalePrice"])
score = cross_val_score(lr, X, y, scoring='mean_squared_error')
# In[*]

print(score)
# In[*]
lr = lr.fit(X, y)
results = lr.predict(test)
final = np.exp(results)

submission = pd.DataFrame()
submission['Id'] = test.index
submission['SalePrice'] = final

submission.to_csv("submission1.csv", index= False)




