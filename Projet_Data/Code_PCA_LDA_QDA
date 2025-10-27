import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.metrics import roc_curve, auc, roc_auc_score

#Importer les données dans un format exploitable
data = pd.read_csv("wdbc.data",header=None)
#print(data.describe()) #On remarque qu'il n'y a pas les noms des colonnes, ajout de header=none
#Il y a également un colonne avec de très grandes valeurs, les ID, et une autre qui n'apparaît pas dans le describe, la variable à valeurs dans {M,B}.

#Mettons d'abord les colonnes 'uniques' (sans _worst _mean _se)
noms_colonnes = ['ID', 'Diagnostique']

#Import du reste 
NomsSans_ = ['radius', 'texture', 'perimeter', 'area', 'smoothness', 'compactness', 'concavity', 'concave_points', 'symmetry', 'fractal_dimension']
NomsAvec_ = []
for e in NomsSans_:
    NomsAvec_.append(e+'_mean')
    NomsAvec_.append(e+'_se')
    NomsAvec_.append(e+'_worst')

for f in NomsAvec_:
    noms_colonnes.append(f)

data.columns = noms_colonnes

#La variable Diagnostique ne prenant que deux valeurs et étant la variable à prédire, il est raisonable de la transformer en une variable à valeurs dans {0,1}.
def diag_to_bin(letter):
    #letter (str) : lettre M ou B
    #return (int) : 1 ou 0 selon la valeur de letter
    if letter=="M":
        return 1
    else: #implicitement "B" sinon elif marche aussi
        return 0
    
diag_bin = data["Diagnostique"].apply(diag_to_bin) #apply pour toutes les valeurs (https://saturncloud.io/blog/pandas-how-to-change-all-the-values-of-a-column/)
data["Diagnostique"]= diag_bin

print(data.describe())
#On remarque qu'environ 37% des valeurs sont des 1 ("M"), ce qui est raisonnable et ne nécessite sans doute pas d'oversampling.

#premiere visualisation
#Matrice de correlations : Traçons la matrice de correlations, celle-ci étant symétrique, pour ne pas surcharger l'illustration, traçons uniquement la partie triangulaire inférieur de cette matrice.

data_sans_id = data.drop(["ID"],axis=1) #On retire l'ID car sa valeur n'a pas d'importance numérique
Matrice_Corr = data_sans_id.corr()
#Création d'un mask pour avoir la partie triangulaire inférieure uniquement (https://stackoverflow.com/questions/57414771/how-to-plot-only-the-lower-triangle-of-a-seaborn-heatmap)
masque = np.triu(Matrice_Corr)

plt.figure(figsize=(12, 10))
sns.heatmap(Matrice_Corr, annot=False,linewidths=1, cmap='coolwarm',mask=masque)
plt.title('Matrice de corrélation')
plt.tight_layout()
plt.savefig('code_PCA_LDA_correlation_matrix.png')
plt.show()


#Réduction de dimensions (Inspiration TD1)

X = data_sans_id
X_sd = StandardScaler().fit_transform(X)

pca = PCA()
pca.fit(X_sd)

# Calcul des valeurs propres
print(pca.explained_variance_)

# Part de variance expliquee
pca.explained_variance_ / pca.explained_variance_.sum()
(pca.explained_variance_ / pca.explained_variance_.sum()).cumsum()

plt.plot(pca.explained_variance_  / pca.explained_variance_.sum())
plt.axhline(1 / X_sd.shape[1], color='k',label="V ariance moyenne théorique 1/p ="+str(np.round((1 / X_sd.shape[1]),decimals=2)))
plt.title("Part de n axe à l'explication de la variance")
plt.legend()
plt.show()

nb_dim_opti = np.argmax(np.cumsum(pca.explained_variance_ratio_) >= 0.9) + 1
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.axvline(nb_dim_opti, color='r',label="Nombre d'axe optimal pour expliquer 90% de la variance (="+str(nb_dim_opti)+")")
plt.title("Part de variance expliquée par n axes")
plt.legend()
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

#LDA et QDA

lda = LinearDiscriminantAnalysis()
lda.fit(data_sans_id.drop('Diagnostique', axis = 1), data_sans_id.Diagnostique)

#Analyse et comparaison des modèles
y = data_sans_id["Diagnostique"]
X = data_sans_id.drop("Diagnostique",axis=1)
X_sd = StandardScaler().fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(
    X_sd, y, test_size=0.3, random_state=42, stratify=y
)

# Fonction pour évaluer les modèles, framework robuste selon le modele utilisé
def evaluation_model(model, X_train, X_test, y_train, y_test):
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    train_score = accuracy_score(y_train, y_train_pred)
    test_score = accuracy_score(y_test, y_test_pred)

    conf_matrix = confusion_matrix(y_test, y_test_pred) # Matrice de confusion
    
    print(conf_matrix)
    
    print(classification_report(y_test, y_test_pred, target_names=['Benign', 'Malignant']))
    
    return {'train_score': train_score,'test_score': test_score,'confusion_matrix': conf_matrix,'model': model} #Valeurs dans le dictionnaire

results = {} # Pour relier les résultats à un modèle choisi.

lda = LinearDiscriminantAnalysis(solver='svd')
lda.fit(X_train, y_train)

results['LDA'] = evaluation_model(lda, X_train, X_test, y_train, y_test)


qda = QuadraticDiscriminantAnalysis()
qda.fit(X_train, y_train)

results['QDA'] = evaluation_model(qda,X_train, X_test, y_train, y_test)

print(results)