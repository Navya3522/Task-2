#!/usr/bin/env python
# coding: utf-8

# # GRIP:- The Spark Foundations
# ## Data Science and Business Analytics - 
# ## Internship Creator:- Navya Shree 
# ## Task-2: Prediction using unsupervised ML 
# ## 2.From the given dataset  "Iris" we have to predict the optimum number of clusters and represent the data visually.

# In[1]:


import numpy as n
import pandas as pd
import matplotlib.pyplot as plt


# In[2]:


data = pd.read_csv('Iris.csv')
data.head()


# In[3]:


data.drop(['Species','Id'],axis=1)


# In[4]:


x = data.iloc[:, [0, 1, 2, 3]].values


# In[5]:


from sklearn.cluster import KMeans


# In[6]:


sse = []

for k in range(1,11):
    km = KMeans(n_clusters=k,max_iter=300,n_init=10,init = 'k-means++')
    km.fit(x)
    sse.append(km.inertia_)
    
plt.plot(range(1,11),sse)
plt.title('The elbow method')
plt.xlabel('Number of clusters')
plt.ylabel('SSE')
plt.show()


# In[7]:


km = KMeans(n_clusters=3,max_iter=300,n_init=10)
y_kmeans = km.fit_predict(x)


# In[8]:


y_kmeans


# In[10]:


plt.scatter(x[y_kmeans == 0,0], x[y_kmeans == 0,1],s=100,c = 'red',label = 'Iris-setosa')

plt.scatter(x[y_kmeans == 1,0], x[y_kmeans == 1,1],s=100,c = 'blue',label = 'Iris-versicolour')

plt.scatter(x[y_kmeans == 2,0], x[y_kmeans == 2,1],s=100,c = 'green',label = 'Iris-virginica')

plt.scatter(km.cluster_centers_[:,0],km.cluster_centers_[:,1], s=100,c='yellow',label='Centroids')

plt.rcParams["figure.figsize"]=10,8


# In[ ]:




