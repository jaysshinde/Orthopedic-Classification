#!/usr/bin/env python
# coding: utf-8

# In[29]:


import pandas as pd


# ### Input files

# In[30]:


data = pd.read_csv('column_2C_weka.csv')
data.tail(10)
x, y = data.loc[:,data.columns!='class'],data.loc[:,'class']
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.5, random_state = 1)


# ### KNN K = 3

# In[31]:


from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors = 3)
x,y = data.loc[:,data.columns != 'class'], data.loc[:,'class']

knn.fit(x_train, y_train)
prediction = knn.predict(x_test)
print('KNN with k = 3 Accuracy = ',knn.score(x_test,y_test))


# ### Varying K to get best Accuracy

# In[36]:


max = 0
k_val = 1
for i in range(1,100):
    k = i
    knn = KNeighborsClassifier(n_neighbors = k)
    knn.fit(x_train, y_train)
    prediction = knn.predict(x_test)
    #print(k , knn.score(x_test, y_test))
    if (knn.score(x_test, y_test) > max):
        max = knn.score(x_test,y_test)
        k_val = k
print('KNN with k = ',k_val,'has Maximum accuracy with ',max)
    


# ### Classification using Decision Tree

# In[35]:


from sklearn import tree
clsfr = tree.DecisionTreeClassifier()
clsfr = clsfr.fit(x_train, y_train)
pred = clsfr.predict(x_test)
print('Accuracy with Decision Tree = ',clsfr.score(x_test,y_test))


# In[ ]:





# In[ ]:




