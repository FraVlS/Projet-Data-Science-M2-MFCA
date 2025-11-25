import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


dat = pd.read_csv("spam.csv", index_col = 0)
print(dat.shape)
dat

#X = dat.drop([ 'Unnamed: 0','type'], axis=1) pour enlever la premiere colonne
X = dat.drop(['type'], axis=1)
y = dat['type']

print(y.unique())
print((y == 'spam').mean())

plt.figure()
plt.matshow(np.corrcoef(np.transpose(X)))
plt.colorbar()
plt.show()

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
(y_train == 'spam').mean(), (y_test == 'spam').mean()

from sklearn.tree import DecisionTreeClassifier, plot_tree
#https://scikit-learn.org/1.5/modules/generated/sklearn.tree.DecisionTreeClassifier.html#sklearn.tree.DecisionTreeClassifier

# par défaut c'est l'algorithme CART qui est implémenté
tree = DecisionTreeClassifier(max_depth=30) 
tree.fit(X_train, y_train)

from sklearn.metrics import confusion_matrix
import seaborn as sns
conf = confusion_matrix(y_test, tree.predict(X_test))

plt.figure()
sns.heatmap(conf, annot=True)
plt.show()

plt.figure()
# jouer avec fontsize pour mieux lire les labels de l'arbre.
plot_tree(tree, max_depth=2, feature_names=X.columns, class_names=tree.classes_, filled=True, fontsize=8)
plt.show()

(y_train == 'spam').sum()
P_test = [] 
P_train = []
for i in np.arange(1, 41):
    tree = DecisionTreeClassifier(max_depth = i)
    tree.fit(X_train, y_train)
    P_test.append((y_test != tree.predict(X_test)).mean())
    P_train.append((y_train != tree.predict(X_train)).mean())

plt.figure()
plt.plot(np.arange(1,41), P_test, 'r*')  
plt.show()

plt.plot(np.arange(1,41), P_train, 'b*') 
plt.show()


#Pour regarder les différences entre deux itérés
plt.plot(np.arange(1,30), np.diff(np.array(P_test)), 'r*')  
plt.plot(np.arange(1,30), np.diff(np.array(P_train)), 'b*')  
plt.plot((0,30),(0,0), '-')
plt.show()


from sklearn.model_selection import GridSearchCV
parameters = {'max_depth': np.arange(1,31)}

tree = DecisionTreeClassifier()
clf = GridSearchCV(tree, parameters, cv = 10)
clf.fit(X_train, y_train)

clf.best_estimator_
clf.cv_results_['mean_test_score']

plt.plot(np.arange(1,31), 1-clf.cv_results_['mean_test_score'])
plt.show()


from sklearn.metrics import confusion_matrix
import seaborn as sns
conf = confusion_matrix(y_test, clf.predict(X_test))
sns.heatmap(conf, annot=True)
plt.show()

(clf.predict(X_test)!= y_test).mean()

conf = confusion_matrix(y_test, clf.predict(X_test))
sns.heatmap(conf, annot=True)


# # Comparaison avec la régression Logit
from sklearn.linear_model import LogisticRegression
SK_logit = LogisticRegression(max_iter=5000, penalty=None)
SK_logit.fit(X_train, y_train)

print((np.exp(SK_logit.coef_)))
conf = confusion_matrix(y_test, SK_logit.predict(X_test))
sns.heatmap(conf, annot=True)
plt.show()

# ## Courbes ROC
from sklearn.metrics import roc_curve, auc

probas = clf.predict_proba(X_test)
fpr0, tpr0, thresholds0 = roc_curve(y_test, probas[:, 0],pos_label = clf.classes_[0] ,  drop_intermediate=False)
fpr0.shape
clf.classes_
fpr0

probas = SK_logit.predict_proba(X_test)
fpr1, tpr1, thresholds0 = roc_curve(y_test, probas[:, 0],pos_label = SK_logit.classes_[0] ,  drop_intermediate=False)
fpr1.shape

auc_tree = auc(fpr0, tpr0) 
auc_logit = auc(fpr1, tpr1)

fig, ax = plt.subplots(1, 1, figsize=(4,4))
ax.plot([0, 1], [0, 1], 'k--')
ax.plot(fpr0, tpr0, label='CART auc=%1.5f' % auc_tree)
ax.set_title('Courbe ROC - spam')
ax.plot(fpr1, tpr1, label='Logit auc=%1.5f' % auc_logit)
ax.set_xlabel("FPR")
ax.set_ylabel("TPR")
ax.legend() 
plt.show()

# # Random forests
# On regarde l'influence de la taille de la forêt
from sklearn.ensemble import RandomForestClassifier
forest1 = RandomForestClassifier(n_estimators=5,oob_score=True)
forest2 = RandomForestClassifier(n_estimators=50,oob_score=True)
forest3 = RandomForestClassifier(n_estimators=200,oob_score=True)
forest1.fit(X_train,y_train)
forest2.fit(X_train,y_train)
forest3.fit(X_train,y_train)

print(1-forest1.oob_score_)
print(1-forest2.oob_score_)
print(1-forest3.oob_score_)


probas = forest1.predict_proba(X_test)
fpr21, tpr21, thresholds21 = roc_curve(y_test, probas[:, 0], pos_label=forest1.classes_[0], drop_intermediate=False)
probas = forest2.predict_proba(X_test)
fpr22, tpr22, thresholds22 = roc_curve(y_test, probas[:, 0], pos_label=forest2.classes_[0], drop_intermediate=False)
probas = forest3.predict_proba(X_test)
fpr23, tpr23, thresholds23 = roc_curve(y_test, probas[:, 0], pos_label=forest3.classes_[0], drop_intermediate=False)


auc_for1 = auc(fpr21, tpr21) 
auc_for2 = auc(fpr22, tpr22) 
auc_for3 = auc(fpr23, tpr23) 


fig, ax = plt.subplots(1, 1, figsize=(4,4))
ax.plot([0, 1], [0, 1], 'k--')
ax.plot(fpr0, tpr0, label='CART auc=%1.5f' % auc_tree)
ax.plot(fpr1, tpr1, label='Logit auc=%1.5f' % auc_logit)
ax.plot(fpr21, tpr21, label= 'Forest 5 auc=%1.5f' % auc_for1)
ax.plot(fpr22, tpr22, label= 'Forest 50 auc=%1.5f' % auc_for2)
ax.plot(fpr23, tpr23, label= 'Forest 200 auc=%1.5f' % auc_for3)
ax.set_title('Courbe ROC - spam')

ax.set_xlabel("FPR")
ax.set_ylabel("TPR")
ax.legend()
plt.show()

(y_test!=forest3.predict(X_test)).mean()


# On remarque qu'il y a un réel gain à considérer une foret: 
# La gain de 1 arbre à 5 arbre (bleu contre vert) est très important
# De même celui du passage de 5 à 50 est lui aussi important
# Ensuite on assiste à une stabilisation: augmenter de 50 à 200 arbres améliore les résultats mais beaucoup
# plus marginalement.
# 
# On peut étudier l'influence de la profondeur de l'arbre : l'algorithme offre de meileurs résultats
# sur des arbres profonds (dont le biais est faible) comparé à des arbres de faibles profondeurs.

# ## Bagging
from sklearn.ensemble import BaggingClassifier
# arbre peu profond
clfA = BaggingClassifier(estimator=DecisionTreeClassifier(max_depth = 5), n_estimators=100)
clfA.fit(X_train, y_train)

conf = confusion_matrix(y_test, clfA.predict(X_test))
sns.heatmap(conf, annot=True)

(y_test!=clfA.predict(X_test)).mean()

# arbre  profond
clfA = BaggingClassifier(n_estimators=100)
clfA.fit(X_train, y_train)

conf = confusion_matrix(y_test, clfA.predict(X_test))
sns.heatmap(conf, annot=True)

(y_test!=clfA.predict(X_test)).mean()

probas = clfA.predict_proba(X_test)
fpr3, tpr3, thresholds3 = roc_curve(y_test, probas[:, 0], pos_label=clfA.classes_[0], drop_intermediate=False)


auc_bag = auc(fpr3, tpr3) 


# # BOOSTING
# Le boosting s'utilise comme le bagging, mais se base cette fois sur des estimateur 
# biaisés (arbre peu profond). En pratique ici le choix de la profondeur peut donner
# des résultats sensiblement différents. On peut donc faire ici une recharche de profondeur
# optimale par VC avec GridSearchCV.
from sklearn.ensemble import AdaBoostClassifier
clfB30 = AdaBoostClassifier(estimator = DecisionTreeClassifier(max_depth=30), n_estimators=50,algorithm='SAMME')
clfB30.fit(X_train, y_train)

clfB = AdaBoostClassifier(estimator = DecisionTreeClassifier(max_depth=1), n_estimators=50,algorithm='SAMME')
clfB.fit(X_train, y_train)

(clfB30.predict(X_test)!= y_test).mean()
(clfB.predict(X_test)!= y_test).mean()


conf = confusion_matrix(y_test, clfB.predict(X_test))
sns.heatmap(conf, annot=True)
probas = clfB.predict_proba(X_test)
fpr4, tpr4, thresholds4 = roc_curve(y_test, probas[:, 0], pos_label=clfB.classes_[0], drop_intermediate=False)
auc_boost = auc(fpr4, tpr4) 


fig, ax = plt.subplots(1, 1, figsize=(4,4))
ax.plot([0, 1], [0, 1], 'k--')

ax.plot(fpr1, tpr1, label='Logit auc=%1.5f' % auc_logit)
ax.plot(fpr23, tpr23, label= 'Forest auc=%1.5f' % auc_for3)
ax.plot(fpr3, tpr3, label= 'Bagging auc=%1.5f' % auc_bag)
ax.plot(fpr4, tpr4, label= 'Boosting auc=%1.5f' % auc_boost)
ax.set_title('Courbe ROC - spam')

ax.set_xlabel("FPR")
ax.set_ylabel("TPR")
ax.legend()


# # XGboost

import xgboost as xgb

label_train = (y_train == 'spam')
label_test = (y_test == 'spam')

dtrain = xgb.DMatrix(X_train, label=label_train)
dtest = xgb.DMatrix(X_test, label=label_test)

param = {'max_depth': 30, 'eta': 0.3, 'objective': 'binary:logistic'}
param['nthread'] = 4
param['eval_metric'] = 'auc'
num_round = 20
evallist = [(dtrain, 'train'), (dtest, 'eval')]

bst = xgb.train(param, dtrain, num_round, evallist)

((bst.predict(dtest)>=0.5)!=(y_test=="spam")).mean()

