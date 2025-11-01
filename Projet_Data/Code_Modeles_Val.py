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

#On choisit un random_state qu'on pourra changer facilement afin de voir si les résultats changent
rand_st=42

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

y = data_sans_id["Diagnostique"]
X = data_sans_id.drop("Diagnostique", axis=1)
X_sd = StandardScaler().fit_transform(X)
pca = PCA()
pca.fit(X_sd)

# Calcul des valeurs propres
print(pca.explained_variance_)

# Part de variance expliquee
pca.explained_variance_ / pca.explained_variance_.sum()
(pca.explained_variance_ / pca.explained_variance_.sum()).cumsum()

#Plot de la variance expliquée vs le nombre d'axes PCA.
plt.plot(pca.explained_variance_  / pca.explained_variance_.sum())
plt.axhline(1 / X_sd.shape[1], color='k',label="V ariance moyenne théorique 1/p ="+str(np.round((1 / X_sd.shape[1]),decimals=2)))
plt.title("Part de n axe à l'explication de la variance")
plt.legend()
plt.savefig('Part de n axe à l explication de la variance.png')
plt.show()

nb_dim_opti = np.argmax(np.cumsum(pca.explained_variance_ratio_) >= 0.9) + 1
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.axvline(nb_dim_opti, color='r',label="Nombre d'axe optimal pour expliquer 90% de la variance (="+str(nb_dim_opti)+")")
plt.title("Part de variance expliquée par n axes")
plt.savefig('Part de variance expliquée par n axes')
plt.show()



# On va maintenant projeter les patients sur les axes PC1 et PC2 afin de voir à quel point ces deux axes séparent les données
X_pca = pca.transform(X_sd)

# Créations de masques pour le scatter
mask_malignant = (y == 1)
mask_benign = (y == 0)

plt.figure(figsize=(8, 6))
plt.scatter(X_pca[mask_benign, 0], X_pca[mask_benign, 1],color='green', alpha=0.5, label='B')
plt.scatter(X_pca[mask_malignant, 0], X_pca[mask_malignant, 1],color='red', alpha=0.5, label='M')
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.legend()
plt.tight_layout()
plt.savefig("Projection des patients sur le plan des deux premiers axes PCA.png")
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
plt.savefig('Correlation Circle Plot axe 1&2.png')
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
plt.savefig('Correlation Circle Plot axe 2&3.png')
plt.show()

#Analyse et comparaison des modèles
y = data_sans_id["Diagnostique"]
X = data_sans_id.drop("Diagnostique",axis=1)
X_sd = StandardScaler().fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_sd, y, test_size=0.2, random_state=rand_st, stratify=y,shuffle=True) #stratify par y pour garder environ 37% de positifs (https://stackoverflow.com/questions/34842405/parameter-stratify-from-method-train-test-split-scikit-learn)

# Fonction pour évaluer les modèles, framework robuste selon le modele utilisé
def evaluation_modele(model, X_train, X_test, y_train, y_test):
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    train_score = accuracy_score(y_train, y_train_pred)
    test_score = accuracy_score(y_test, y_test_pred)

    conf_matrix = confusion_matrix(y_test, y_test_pred) # Matrice de confusion
    print(model)
    print(conf_matrix)
    
    print(classification_report(y_test, y_test_pred, target_names=['Benign', 'Malignant']))
    
    return {'train_score': train_score,'test_score': test_score,'confusion_matrix': conf_matrix,'model': model} #Valeurs dans le dictionnaire

results = {} # Pour relier les résultats à un modèle choisi.


#LDA

lda = LinearDiscriminantAnalysis(solver='svd')
lda.fit(X_train, y_train)
results['LDA'] = evaluation_modele(lda, X_train, X_test, y_train, y_test)


#QDA

qda = QuadraticDiscriminantAnalysis()
qda.fit(X_train, y_train)
results['QDA'] = evaluation_modele(qda,X_train, X_test, y_train, y_test)

#Logistique
from sklearn.linear_model import LogisticRegression

logit = LogisticRegression(max_iter=2000, random_state=rand_st, class_weight='balanced')
logit.fit(X_train, y_train)
results['Logistic Regression'] = evaluation_modele(
    logit, X_train, X_test, y_train, y_test
)

#KNN
from sklearn.neighbors import KNeighborsClassifier

param_grid_knn = {'n_neighbors': [3, 5, 7, 9, 11, 13, 15, 20, 25]}
knn_base = KNeighborsClassifier()
knn_grid = GridSearchCV(knn_base, param_grid_knn, cv=5, scoring='recall', n_jobs=-1, verbose=1)
knn_grid.fit(X_train, y_train)
print("Meilleur k: "+ str(knn_grid.best_params_['n_neighbors']))
print("Score validation croisée: " + str(knn_grid.best_score_))
results['KNN'] = evaluation_modele(knn_grid.best_estimator_,X_train, X_test, y_train, y_test)

#RFC
from sklearn.ensemble import RandomForestClassifier

param_grid_rf = {'n_estimators': [50, 100, 200],'max_depth': [5, 10, 15, None],'min_samples_split': [2, 5, 10]}
rf_base = RandomForestClassifier(random_state=rand_st, class_weight='balanced')
rf_grid = GridSearchCV(rf_base, param_grid_rf, cv=5, scoring='recall',n_jobs=-1, verbose=1)
rf_grid.fit(X_train, y_train)
print("Meilleur k: "+ str(rf_grid.best_params_))
print("Score validation croisée: " + str(rf_grid.best_score_))
results['Random Forest'] = evaluation_modele(rf_grid.best_estimator_, X_train, X_test, y_train, y_test)

# Stockage des features_importance du Random Forest en dictionnaire pour le wordcloud final
features_importance_rf = dict(zip(X.columns, rf_grid.best_estimator_.feature_importances_))

#Comparaison : 

# Courbes ROC

plt.figure(figsize=(8, 6))
plt.plot([0, 1], [0, 1], color="k",label='Classifieur 50/50 (indépendant des données)')

for name, res in results.items():
    model = res['model']
    y_pred = model.predict_proba(X_test)[:, 1]
    faux_pos, vrai_pos, thresholds = roc_curve(y_test, y_pred)
    plt.plot(faux_pos, vrai_pos, label = name + " (AUC=" + str(np.round(roc_auc_score(y_test, y_pred), 3)) + ")")

plt.title("Courbes ROC")
plt.xlabel("Taux de Faux Positifs")
plt.ylabel("Taux de Vrais Positifs")
plt.legend()
plt.savefig('Courbes ROC.png')
plt.show()

# AUC 
print("AUC par algorithme : ")
for name, res in results.items():
    model = res['model']
    y_pred = model.predict_proba(X_test)[:, 1]
    print(name +"    "+ str(roc_auc_score(y_test, y_pred)))


# Modèle Logistique avec PCA
X_features = data_sans_id.drop("Diagnostique", axis=1)
y = data_sans_id["Diagnostique"]

X_sd = StandardScaler().fit_transform(X_features)
pca = PCA()
pca.fit(X_sd)

# Graphique des coefficients PCA 
PC = pca.components_.T * np.sqrt(pca.explained_variance_)
PC_df = pd.DataFrame(PC[:, :2], index=X_features.columns, columns=['PC1', 'PC2']) #Création du DF avec les coeffs

PC1_sorted = PC_df['PC1'].sort_values(ascending=False) #Ordre décroissant pour PC1
PC2_sorted = PC_df['PC2'].sort_values(ascending=False) #Ordre décroissant pour PC2

plt.figure(figsize=(10,5))
sns.barplot(x=PC1_sorted.index, y=PC1_sorted.values, color='red') #Barplot pour les coeffs
plt.xticks(rotation=60)
plt.title("Contributions des variables à l'axe 1")
plt.ylabel("Coeff PC1")
plt.tight_layout()  #Sinon les noms des variables tiennent pas dans le cadre
plt.savefig("Contributions des variables à l'axe 1.png")
plt.show()

plt.figure(figsize=(10,5))
sns.barplot(x=PC2_sorted.index, y=PC2_sorted.values, color='blue')
plt.xticks(rotation=60)
plt.title("Contributions des variables à l'axe 2")
plt.ylabel("Coeff PC2")
plt.tight_layout() #Sinon les noms des variables tiennent pas dans le cadre
plt.savefig("Contributions des variables à l'axe 2.png")
plt.show()

# Test de la Régression Logistique sur un nombre n variable d'axes de la PCA

X_pca = pca.transform(X_sd)
X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.2, random_state=rand_st, stratify=y)


nb_components = X_pca.shape[1] #Nombre
range_compon = range(1, nb_components + 1)

auc_list=[]
aic_list=[]
bic_list=[]

n = X_test.shape[0] #Nombre d'observations pour bic

from sklearn.metrics import log_loss
for n_components in range_compon:
    X_train_pca = X_train[:, :n_components] #Garder les n premieres composantes
    X_test_pca = X_test[:, :n_components] #Idem

    logit = LogisticRegression(max_iter=2000, random_state=rand_st,class_weight='balanced')
    logit.fit(X_train_pca, y_train)
    
    y_pred_proba = logit.predict_proba(X_test_pca)[:,1] #Proba de classe 1
    auc_score = roc_auc_score(y_test, y_pred_proba)
    
    L = -log_loss(y_test, y_pred_proba, normalize=False)
    k = n_components+1
    aic = 2*k - 2*L #Formule du cours
    bic = np.log(n)*k-2*L #Formule du cours

    auc_list.append(auc_score)
    aic_list.append(aic)
    bic_list.append(bic)

plt.figure(figsize=(10,6))
plt.plot(range_compon, auc_list, label='AUC', color='blue')
plt.title("AUC en fonction du nombre d'axes PCA")
plt.xlabel("Nombre d'axes")
plt.ylabel("AUC")
plt.legend()
plt.savefig("AUC en fonction du nombre d'axes PCA.png")
plt.show()

plt.figure(figsize=(10,6))
plt.plot(range_compon, aic_list, label='AIC', color='red')
plt.plot(range_compon, bic_list, label='BIC', color='green')
plt.title("AIC et BIC selon le nombre d'axes PCA")
plt.xlabel("Nombre d'axes")
plt.ylabel("Valeur du critère")
plt.legend()
plt.savefig("AIC et BIC selon le nombre d'axes PCA.png")
plt.show()

# Évaluation du modèle Logistique sur 5 axes PCA. Ce n'est pas le meilleur pour le random_state=42 mais après avoir d'autres random_states, la valeur n=5 revient plus souvent pour le min de l'AIC et BIC.

X_train_pca5 = X_train[:, :5] # Sélection des 5 premières composantes
X_test_pca5 = X_test[:, :5] #idem

logit_pca4 = LogisticRegression(max_iter=2000, random_state=rand_st,class_weight='balanced')
logit_pca4.fit(X_train_pca5, y_train)

from sklearn.metrics import recall_score, precision_score

# threshold qui fait en sorte d'avoir 0 faux négatifs.
y_proba = logit_pca4.predict_proba(X_test_pca5)[:, 1]
threshold = 0.3
y_pred_thr = (y_proba >= threshold).astype(int)

cm = confusion_matrix(y_test, y_pred_thr)
recall = recall_score(y_test, y_pred_thr)
precision = precision_score(y_test, y_pred_thr)
specificity = recall_score(y_test, y_pred_thr, pos_label=0)


print("Matrice de confusion avec t="+str(threshold)+": ", cm)


res_logit_pca4 = evaluation_modele(logit_pca4, X_train_pca5, X_test_pca5, y_train, y_test)


# AUC & courbe ROC
y_pred_proba_pca4 = logit_pca4.predict_proba(X_test_pca5)[:, 1]
auc_pca4 = roc_auc_score(y_test, y_pred_proba_pca4)
print("AUC pour n=5 axes : "+ str(np.round(auc_pca4,3)))

# Tracé de la courbe ROC
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba_pca4)
plt.figure(figsize=(7, 6))
plt.plot(fpr, tpr, color='orange', lw=2,label='Logistique (5 composantes) (AUC = '+str(np.round(auc_pca4,3))+')')
plt.plot([0, 1], [0, 1], color='grey', linestyle='--')
plt.title("Courbe ROC de la Régression Logistique sur 5 axes de la PCA")
plt.xlabel("Taux de Faux Positifs")
plt.ylabel("Taux de Vrais Positifs")
plt.legend()
plt.savefig("Courbe ROC de la Régression Logistique sur 5 axes de la PCA.png")
plt.show()

# Nuage de mots pour illustrer les éléments les plus importants du PC1 sachant qu'ils sont tous de coeff positifs.

from wordcloud import WordCloud
freq_PC1 = PC_df['PC1'].abs().to_dict()
wc_PC1 = WordCloud(width=800, height=400, background_color='white',colormap='copper').generate_from_frequencies(freq_PC1)

plt.figure(figsize=(16, 8))
plt.imshow(wc_PC1, interpolation='bilinear') #(https://www.geeksforgeeks.org/python/generating-word-cloud-python/)
plt.title('Nuage de mots PC1')
plt.axis('off') #Pas besoin d'axes
plt.tight_layout()
plt.savefig('Nuage de mots PC1.png')
plt.show()

# Nuage de mots pour illustrer les features les plus importantes de RF

wc_RF = WordCloud(width=800, height=400, background_color='white',colormap='copper').generate_from_frequencies(features_importance_rf)
plt.figure(figsize=(16, 8))
plt.imshow(wc_RF, interpolation='bilinear')
plt.title('Nuage de mots RF')
plt.axis('off') #Pas besoin d'axes
plt.tight_layout()
plt.savefig('Nuage de Mots RF.png')
plt.show()
