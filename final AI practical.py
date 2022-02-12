#!/usr/bin/env python
# coding: utf-8

# In[1]:


# lcm gcd
def gcd(a,b):
    if a==0:
        return b
    return gcd(b%a,a)

def lcm(a,b):
    return(a/gcd(a,b))*b

n1=12
n2=60
print(lcm(n1,n2))
print(gcd(n1,n2))


# In[9]:


# bfs
graph={
    '5':['3','7'],
    '3':['2','4'],
    '7':['8'],
    '2':[],
    '4':['8'],
    '8':[]
}

visited=[]
queue=[]

def bfs(visited,graph,node):
    visited.append(node)
    queue.append(node)
    while queue:
        m=queue.pop(0)
        print(m)
        for neighbour in graph[m]:
            bfs(visited,graph,neighbour)
            
 
bfs(visited,graph,'5')


# In[11]:


# dfs
graph={
    '5':['3','7'],
    '3':['2','4'],
    '7':['8'],
    '2':[],
    '4':['8'],
    '8':[]
}

visited=set()

def dfs(visited,graph,node):
    if node not in visited:
        visited.add(node)
        print(node)
        
        for neighbour in graph[node]:
            dfs(visited,graph,neighbour)

dfs(visited,graph,'5')


# In[ ]:


from collections import defaultdict
  
jug1, jug2, aim = 4, 3, 2
  
visited = defaultdict(lambda: False)
  
def waterJugSolver(amt1, amt2): 
    if (amt1 == aim and amt2 == 0) or (amt2 == aim and amt1 == 0):
        print(amt1, amt2)
        return True
    if visited[(amt1, amt2)] == False:
        print(amt1, amt2)
        
        visited[(amt1, amt2)] = True
      
        return (waterJugSolver(0, amt2) or
                waterJugSolver(amt1, 0) or
                waterJugSolver(jug1, amt2) or
                waterJugSolver(amt1, jug2) or
                waterJugSolver(amt1 + min(amt2, (jug1-amt1)),
                amt2 - min(amt2, (jug1-amt1))) or
                waterJugSolver(amt1 - min(amt1, (jug2-amt2)),
                amt2 + min(amt1, (jug2-amt2))))
      
    else:
        return False
  
print("Steps: ")
  
waterJugSolver(0, 0)


# In[4]:


# knn
import numpy as np
import pandas as pd

df_iris=pd.read_csv("Iris.csv")
df_iris.head()

X_response=df_iris.iloc[:,1:-1]
Y_target=df_iris.iloc[:,-1:]

from sklearn.model_selection import train_test_split

X_train,X_test,Y_train,Y_test=train_test_split(X_response,Y_target,test_size=0.2)

from sklearn.preprocessing import StandardScaler

scaler=StandardScaler()

scaler.fit(X_train)
X_train=scaler.transform(X_train)
X_test=scaler.transform(X_test)

from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train,Y_train)

Y_pred=knn.predict(X_test)

from sklearn.metrics import accuracy_score
accuracy_score(Y_test,Y_pred)


# In[20]:


# regression algorithm
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns;sns.set()

from sklearn.datasets import make_blobs

X, y = make_blobs(n_samples=300, centers=2, random_state=2, cluster_std=1.5)
plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='summer');
plt.show()

from sklearn.naive_bayes import GaussianNB
model_GNB = GaussianNB()
model_GNB.fit(X, y)

rng = np.random.RandomState(0)
Xnew = [-6, -14] + [14, 18] * rng.rand(2000, 2)
ynew = model_GNB.predict(Xnew)

plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='summer')
lim = plt.axis()
plt.scatter(Xnew[:, 0], Xnew[:, 1], c=ynew, s=20, cmap='summer', alpha=0.1)
plt.axis(lim)
plt.show()

yprob = model_GNB.predict_proba(Xnew)
yprob[-10:].round(3)


# In[22]:


# random forest
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import make_blobs

X,y=make_blobs(n_samples=300,centers=2,random_state=0,cluster_std=0.50)

plt.scatter(X[:,0],X[:,1],c=y,s=50,cmap='summer')
plt.show()

xfit = np.linspace(-1, 3.5)
plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='summer')
plt.plot([0.6], [2.1], 'x', color='black', markeredgewidth=4, markersize=12)
for m, b in [(1, 0.65), (0.5, 1.6), (-0.2, 2.9)]:
    plt.plot(xfit, m * xfit + b, '-k')
plt.xlim(-1, 3.5)
plt.show()

xfit = np.linspace(-1, 3.5)
plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='summer')
for m, b, d in [(1, 0.65, 0.33), (0.5, 1.6, 0.55), (-0.2, 2.9, 0.2)]: 
    yfit = m * xfit + b
    plt.plot(xfit, yfit, '-k')
    plt.fill_between(xfit, yfit - d, yfit + d, edgecolor='none',
    color='b', alpha=0.4)
plt.xlim(-1, 3.5)
plt.show()


# In[ ]:


# simple chat box
from numpy import random
greet={'input':['hi','hello','namaste'],
      'response':['hello','how are you']}
intro={'input':['who are you?','what is your name?'],
      'response':['my name is cool-bot and im chat bot']}

while True:
    user_input=input("User>> ").lower()
    if user_input in greet['input']:
        res=random.choice(greet['response'])
        print(f'Bot>> {res}')
    elif user_input in intro['input']:
        res=random.choice(intro['response'])
        print(f'Bot>> {res}')    
    elif user_input=='bye':
        print("Bot>> bye")
        break
    else:
        print("Bot>> can't understand what are you saying")

