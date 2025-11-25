
import pandas as pd
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt

# importation des données
df = pd.read_csv('decathlon2.dat',index_col=0, delimiter=' ') #
print(df)
df.head()
df.dtypes
df.shape


# Afficher une colonne de plusieurs facons
print(df.c110)
print(df[df.columns[0]])
print(df.iloc[:, 1])

# Calcul des correlations des variables quanti
X = df
X = X.drop(['COMPET'],axis=1)
print(X.corr())
# Sous forme de graphique
plt.matshow(np.corrcoef(X, rowvar = False))
plt.show()

# On utilise cette palette pour avoir des couleurs
# faciles a lire pour la correlation
import seaborn as sns
cmap = sns.diverging_palette(230, 20, as_cmap=True)

# On utilise vmin et vmax pour imposer aux couleurs d'etre 
# entre -1 and +1
sns.heatmap(X.corr(),cmap=cmap,vmin=-1,vmax=1)
plt.show()

# Pour l'ACP, on enleve les variables RANG et POINTS
X = X.drop(['RANG','POINTS'],axis=1)
X.shape
print(X)

# Centrer et reduire X avant de faire l'ACP
from sklearn.preprocessing import StandardScaler
X_sd = StandardScaler().fit_transform(X)

# la moyenne est nulle, l'ecart type vaut 1
X_sd.std(axis=0)
X_sd.mean(axis=0)

# L'ACP
pca = PCA()
pca.fit(X_sd)

# Calcul des valeurs propres
print(pca.explained_variance_)

# Part de variance expliquee
pca.explained_variance_ / pca.explained_variance_.sum()
(pca.explained_variance_ / pca.explained_variance_.sum()).cumsum()

plt.plot(pca.explained_variance_  / pca.explained_variance_.sum(),'b.')
plt.axhline(1 / X_sd.shape[1], color='k', linestyle='--')
plt.show()

# Calcul du cercle des correlations
PC = pca.components_.T*np.sqrt(pca.explained_variance_)
PC
# Pour afficher les valeurs des correlations entre les variables et les axes
# En ligne les variables et en colonne les axes.
pd.DataFrame(PC,index=X.columns)


# Cercle des correlations entre le premier et deuxieme axe
plt.figure(figsize=(8, 8))
plt.title('Correlation Circle Plot')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')

for i in range(0,X.shape[1]):
    plt.arrow(0, 0,PC[i, 0],PC[i, 1],color='black',alpha=0.7,width=0.005)

feature = X.columns
for i in range(0,X.shape[1]):
    plt.annotate(feature[i], (PC[i, 0],PC[i, 1]),color='red')

plt.xlim(-1,1)
plt.ylim(-1,1)
plt.grid(True)
plt.gca().add_artist(plt.Circle((0,0),1,color='blue',fill=False))
plt.show()
# Premier axe : c110 (cor=0.76), c100 (cor=0.78) et c400 (cor=0.69) sont très corrélés (positivement),
# long (cor=-0.75), haut (cor=-0.58), poids (cor=-0.63) et disq (cor=-0.56)  sont très corrélés négativement
# c1500 (cor=0.06), et perch (cor=-0.05), ne sont quasiment pas corrélés au premier axe

# Deuxième axe : disq (cor=0.61), poids (cor=0.61) et c400 (cor=0.58) sont très corrélés (positivement),
# c1500 (cor=0.48), haut (cor=0.35) et javel (cor=0.32) sont moyennement corrélés au deuxième axe
# long (cor=-0.35) est moyennement corrélé (négativement) au deuxième axe

# Cercle des correlations entre le deuxieme et troisieme axe
plt.figure(figsize=(8, 8))
plt.title('Correlation Circle Plot')
plt.xlabel('Principal Component 2')
plt.ylabel('Principal Component 3')

for i in range(0,X.shape[1]):
    plt.arrow(0, 0,PC[i, 1],PC[i, 2],color='black',alpha=0.7,width=0.005)

feature = X.columns
for i in range(0,X.shape[1]):
    plt.annotate(feature[i], (PC[i, 1],PC[i, 2]),color='red')


plt.xlim(-1,1)
plt.ylim(-1,1)
plt.grid(True)
plt.gca().add_artist(plt.Circle((0,0),1,color='blue',fill=False))
plt.show()
# Troisième axe : C1500 (cor=0.79), perche (cor=0.70) sont très corrélés (positivement),
# javel (cor=-0.39), haut (cor=-0.26) sont moyennement corrélés (négativement)


# Contribution des individus aux axes
# La matrice des facteurs s'obtient directement avec la commande suivante

F = pca.transform(X_sd) #pca.transform centre automatiquement les données avant d'appliquer la PCA.
F.shape
pd.DataFrame(F,index=df.index)

# On peut egalement la recalculer en passant par la matrice U
U = pca.components_
# sous Python U est la matrice transposée des
# composantes principales
# U contient les vecteurs des nouvelles coordonnées en ligne
# (au lieu de les avoir en colonne)
# Prendre la transposée pour avoir U dans le "bon sens" (comme dans le cours)
np.matmul(X_sd,U.transpose())
# On retrouve F

# Les contributions des individus aux axes (formule du cours)
Contrib_ind = (F**2)/(pca.explained_variance_*41)
pd.DataFrame(Contrib_ind*100,index=df.index)

# sur l'axe 1, Drews (4.48%), Yurkov (3.86%) et Casarsa (3.47%) sont ceux qui contribuent le plus
# sur l'axe 2, Karpov (14.41%) Bourguignon (11.92%), Sebrle (11.00%) et Smith (7.37%) sont ceux qui contribuent le plus
# sur l'axe 3, Korkizoglou (7.52%) MARTINEAU (6.66%) et Casarsa (6.29%) sont ceux qui contribuent le plus

# Qualité de représentation des individus au premier axe
deno = np.sum(X_sd**2,axis=1)
num=(np.matmul(X_sd,U[0,:])**2)
num/deno

quali = np.zeros((41,3))
# Qualité de représentation des individus au trois premiers axes
for j in range(0,3):
    num=(np.matmul(X_sd,U[j,:])**2)
    quali[:,j]=num/deno

print(quali)
pd.DataFrame(quali,index=df.index)*100
pd.DataFrame(quali.cumsum(axis=1),index=df.index)*100
# Warners est très bien représenté dans le plan factoriel (composé des trois axes), a 97.80%
# BOURGUIGNON egalement, a 95.42%, Casarsa a 93.28% et Drews a 91.10%.
# McMULLEN est tres mal represente par les trois axes (8.73%) !!

#################################################################
# Representation des individus dans les plans factoriels
# On affiche les individus qui contribuent le plus sur chaque axe

# Premier plan factoriel

plt.figure(figsize=(8, 8))
plt.scatter(F[:,0],F[:,1])
for i in range(0,X.shape[0]):
    # annotation uniquement si valeur absolue sur une de 2 dimensions importantes (valeurs choisies empiriquement)
    if (abs(F[i,0]) > 2) | (abs(F[i,1]) > 2):
        plt.annotate(df.index[i],(F[i,0],F[i,1]), color="red")
plt.title('Premier plan factoriel')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.show()
# On retrouve Drews, Yurkov, Casarsa sur le premier axe et Karpov, Bourguignon,
# Sebrle et Smith sur le deuxième axe (voir resultats sur les contributions des
# individus aux axes).

# Deuxième plan factoriel

plt.figure(figsize=(8, 8))
plt.scatter(F[:,1],F[:,2])
for i in range(0,X.shape[0]):
    # annotation uniquement si valeur absolue sur une de 2 dimensions importantes (valeurs choisies empiriquement)
    if (abs(F[i,1]) > 2) | (abs(F[i,2]) > 1.5):
        plt.annotate(df.index[i],(F[i,1],F[i,2]), color="red")
plt.title('Deuxieme plan factoriel')
plt.xlabel('Principal Component 2')
plt.ylabel('Principal Component 3')
plt.show()
# On retrouve Korkizoglou, MARTINEAU et Casarsa sur le troisièle axes (voir resultats sur les contributions des
# individus aux axes).

##########################################################
# Si on multiplie maintenant les donnees de courses par -1
print(df.iloc[:,[0,4,5,9]])
df.iloc[:,[0,4,5,9]] = (df.iloc[:,[0,4,5,9]])*(-1)
print(df)

X2=df
X2 = X2.drop(['COMPET','RANG','POINTS'],axis=1)
X2.shape
print(X2)

# On centre et on reduit
X2_sd = StandardScaler().fit_transform(X2)

# L'ACP
pca = PCA()
pca.fit(X2_sd)

# Calcul des valeurs propres
print(pca.explained_variance_)

# Part de variance expliquee
pca.explained_variance_ / pca.explained_variance_.sum()

plt.plot(pca.explained_variance_  / pca.explained_variance_.sum(),'b.')
plt.axhline(1 / X2_sd.shape[1], color='k', linestyle='--')
plt.show()

PC = pca.components_.T*np.sqrt(pca.explained_variance_)
# Cercle des correlations entre le premier et deuxieme axe
plt.figure(figsize=(8, 8))
plt.title('Correlation Circle Plot')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')

for i in range(0,X2.shape[1]):
    plt.arrow(0, 0,PC[i, 0],PC[i, 1],color='black',alpha=0.7,width=0.005)

feature = X2.columns
for i in range(0,X2.shape[1]):
    plt.annotate(feature[i], (PC[i, 0],PC[i, 1]),color='red')

plt.xlim(-1,1)
plt.ylim(-1,1)
plt.grid(True)
plt.show()

# Cercle des correlations entre le deuxieme et troisieme axe
plt.figure(figsize=(8, 8))
plt.title('Correlation Circle Plot')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')

for i in range(0,X2.shape[1]):
    plt.arrow(0, 0,PC[i, 1],PC[i, 2],color='black',alpha=0.7,width=0.005)

feature = X2.columns
for i in range(0,X2.shape[1]):
    plt.annotate(feature[i], (PC[i, 1],PC[i, 2]),color='red')


plt.xlim(-1,1)
plt.ylim(-1,1)
plt.grid(True)
plt.show()