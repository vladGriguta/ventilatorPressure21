#!/usr/bin/env python
# coding: utf-8

# In[141]:


get_ipython().run_line_magic('cd', '/Users/andrei.toma/Downloads/ventilator-pressure-prediction/')


# In[142]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[164]:


train = pd.read_csv('train.csv')
train.head(50)


# In[166]:


n=1
plt.figure(figsize = (12,5))
sns.scatterplot(y = train[train['breath_id']==n]['pressure']
                ,x = train[train['breath_id']==n]['time_step'])


# In[170]:


plt.figure(figsize = (12,5))
train[train['breath_id']==1]['pressure'].plot(kind = 'line')
train[train['breath_id']==1]['u_in'].plot(kind = 'line')


# In[146]:


y = train['pressure'].values
X = train.drop('pressure',axis = 1).values


# In[147]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping


# In[148]:


# from sklearn.model_selection import train_test_split

# X_train, X_test, y_train, y_test = train_test_split(
#                                         X, y, test_size=0.3, random_state=42)

# print(X_train.shape,y_train.shape)


# In[149]:


X_train = train.iloc[:4225200,:-1].values
X_test = train.iloc[4225200:,:-1].values
y_train = train.iloc[:4225200]['pressure'].values
y_test = train.iloc[4225200:]['pressure'].values


# In[150]:



model_NN = Sequential()

model_NN.add(Dense(X_train.shape[1],activation = 'relu'))
model_NN.add(Dropout(rate = 0.5))

for i in range(5):
    model_NN.add(Dense((X_train.shape[1])/2,activation = 'relu'))
    model_NN.add(Dropout(rate = 0.5))



model_NN.add(Dense(1,activation = 'relu'))

model_NN.compile(loss = 'mean_squared_error',optimizer = 'adam')

early_stop = EarlyStopping(monitor='val_loss',mode = 'min',verbose = 0,patience = 5)

model_NN.fit(x = X_train,y = y_train,epochs=100,validation_data=(X_test,y_test),
             callbacks = [early_stop],verbose = 1)

losses = pd.DataFrame(model_NN.history.history)
losses.plot()
predictions = model_NN.predict(X_test)

from sklearn.metrics import mean_squared_error,mean_absolute_error

print(""" MAE """,mean_absolute_error(y_test,predictions))
print(""" MSE """,mean_squared_error(y_test,predictions))


# In[180]:


train = pd.read_csv('train.csv')
train.head()


# In[181]:


y = train['pressure'].values
X = train.drop(['id','breath_id','pressure'],axis = 1).values


# In[186]:


#perform a split based on breaths. first 70% are assigned to X/y train, and the rest to X/y test
X_train = train.iloc[:4225200,:-1]
X_test = train.iloc[4225200:,:-1]
y_train = train.iloc[:4225200]['pressure']
y_test = train.iloc[4225200:]['pressure']
# .values


# In[ ]:





# In[184]:


#perform a random split
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
                                        X, y, test_size=0.3, random_state=42)


# In[187]:


from sklearn.linear_model import LinearRegression
lm = LinearRegression()

lm.fit(X_train,y_train)
pred = lm.predict(X_test)

print(""" MAE """,mean_absolute_error(y_test,pred))
print(""" MSE """,mean_squared_error(y_test,pred))

