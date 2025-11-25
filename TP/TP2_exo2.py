#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt

import matplotlib.pyplot as plt
import pandas as pd


# In[2]:


dfX = pd.read_csv('digitsX.csv').drop('Unnamed: 0', axis=1)
dfXt = pd.read_csv('digitsXt.csv').drop('Unnamed: 0', axis=1)
dfY = pd.read_csv('digitsY.csv').drop('Unnamed: 0', axis=1)
dfYt = pd.read_csv('digitsYt.csv').drop('Unnamed: 0', axis=1)


# In[5]:


print(dfX)


# In[7]:


print(dfY)


# In[9]:


dfX.shape


# In[11]:


dfX.dtypes


# In[13]:


print(dfX)


# In[21]:


X = dfX.to_numpy()
Y = dfY.to_numpy().reshape(3000,)
Xt = dfXt.to_numpy()
Yt = dfYt.to_numpy().reshape(1500,)

a= 2000
plt.matshow(X[a,:].reshape((28,28)))
Y[a]


# In[19]:


X.shape


# In[41]:


np.unique(Y)
print((Y==1).mean())
print((Y==7).mean())
print((Y==8).mean())


# In[43]:


from sklearn.decomposition import PCA

pca = PCA()
pca.fit(X)
F = pca.transform(X)

plt.scatter(F[:,0], F[:,1], c=Y, cmap=plt.get_cmap("tab10"), s=3)
plt.colorbar()


# On trace les individus dans les 2 premiers plans factoriels pour visualiser le nuage
# dans un repère qui conserve le plus d'information, pour exemple dans un autre plan factoriel
# le nuage des individus est beaucoup moins interprétable: on ne voit rien.

# In[45]:


plt.scatter(F[:,20], F[:,113], c=Y, cmap=plt.get_cmap("tab10"), s=3)


# Dans le premier graphe on remarque que les 3 classes sont relativement bien séparées,
# on en déduit q'une méthode de classification (même simple) devrait permettre de séparer
# facilement les 3 classes, on s'attend à des erreurs de classification faibles comme on l'observe
# ci dessous.
# 
# On reviendra vers ce graphe pour interpréter la matrice de confusion: 
# Les erreurs les plus importantes devraient avoir lieu pour des classes dont la frontière
# est peu nette. 

# In[48]:


from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.metrics import confusion_matrix


lda = LinearDiscriminantAnalysis()
lda.fit(X,Y)


# In[50]:


Predlin = lda.predict(X)# in sample
(Predlin != Y).mean()

Predlin = lda.predict(Xt)# out of sample
(Predlin !=Yt).mean()


# In[52]:


# Matrice confusion:
from sklearn.metrics import confusion_matrix

pd.DataFrame(confusion_matrix( Yt, Predlin), index =['V1', 'V7', 'V8'], columns = ['P1', 'P7', 'P8'])


# On remarque que l'erreur la plus importante (en mouyenne) est que l'algorithme prédit 1
# alors qu'il s'agit d'un 7 ou d'un 8 (première colonne). La classe des 1 est la mieux 
# retrouvée (1 ligne; un vrai 1 est plus rarement classé comme 7 ou 8).

# In[23]:


from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
qda = QuadraticDiscriminantAnalysis()

qda.fit(X,Y)


# In[25]:


# Errors
Predqda = qda.predict(X)# in sample
print((Predqda != Y).mean())

Predqda = qda.predict(Xt)# out of sample
print((Predqda != Yt).mean())


# In[29]:


# Matrice confusion:
from sklearn.metrics import confusion_matrix

pd.DataFrame(confusion_matrix(Yt, Predqda), index =['V1', 'V7', 'V8'], columns = ['P1', 'P7', 'P8'])
# Résultats identiques à la méthode linéaire.


# In[ ]:




