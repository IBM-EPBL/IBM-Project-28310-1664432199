#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[4]:


df1=pd.read_csv('Mall_Customers.csv')
df1.head()


# In[6]:


df1.info()


# In[7]:


df1.describe()


# In[8]:


df1.shape


# Univariate Analysis

# In[9]:


sns.histplot(df['Annual Income (k$)'])


# In[10]:


sns.boxplot(df['Annual Income (k$)'])


# In[11]:


sns.distplot(df['Annual Income (k$)'])


# In[12]:


sns.barplot(df['Age'],df['Annual Income (k$)'])


# In[14]:


sns.lineplot(df['Annual Income (k$)'], df['Spending Score (1-100)'])


# In[15]:


sns.scatterplot(df['Spending Score (1-100)'],df['Age'],hue =df['Gender'])


# Multi-variate Analysis

# In[17]:


sns.pairplot(data=df[["Age", "Gender","Spending Score (1-100)","Annual Income (k$)"]])


# In[18]:


sns.heatmap(df.corr(),annot=True)


# Perform descriptive statistics on the dataset

# In[19]:


df1.describe()


# In[20]:


df1.drop('CustomerID',axis=1,inplace=True)
df1.head()


# Check for the missing values and deal with them

# In[21]:


df.isnull().any()


# Find the outliers and replace the outliers

# In[22]:


sns.boxplot(df['Age'])


# Check for categorical columns and perform encoding

# In[23]:


from sklearn.preprocessing import LabelEncoder
l_en = LabelEncoder()


# In[24]:


df1['Gender'] = l_en.fit_transform(df1['Gender'])
df1.head()


# Scaling the data

# In[26]:


from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(df1)
data_scaled[0:5]


#  Perform any of the clustering algorithm

# In[27]:


from sklearn.cluster import KMeans
km = KMeans()
res = km.fit_predict(data_scaled)
res


# In[29]:


data = pd.DataFrame(data_scaled, columns = df1.columns)
data.head()


# In[30]:


data['kclus']  = pd.Series(res)
data.head()


# In[31]:


data['kclus'].unique()


# In[32]:


data['kclus'].value_counts()


# In[33]:


ind = data.iloc[:,0:4]
ind.head()


# In[34]:


dep = data.iloc[:,4:]
dep.head()


# Split the data into training and testing

# In[35]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(ind,dep,test_size=0.3,random_state=1)
x_train.head()


# In[36]:


x_test.head()


# In[37]:


y_train.head()


# In[38]:


from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(x_train,y_train)


# In[39]:


pred_test = lr.predict(x_test)
pred_test[0:5]


# Measure the performance using metrics

# In[40]:


from sklearn.metrics import mean_squared_error,mean_absolute_error
from sklearn.metrics import accuracy_score
mse = mean_squared_error(pred_test,y_test)
print("The Mean squared error is: ", mse)
rmse = np.sqrt(mse)
print("The Root mean squared error is: ", rmse)
mae = mean_absolute_error(pred_test,y_test)
print("The Mean absolute error is: ", mae)
acc = lr.score(x_test,y_test)
print("The accuracy is: ", acc)


# In[ ]:




