

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression




Y, X = pd.read_csv("Telecom_y.csv", index_col = 0) ,  pd.read_csv("Telecom_x.csv", index_col = 0)



print(Y)


print(X)




X.columns



X.dtypes



Y.mean()




X.mean(axis=0)




X.std()



plt.figure()
plt.matshow(np.corrcoef(X, rowvar = False))
plt.colorbar()
plt.show()



# Moyennes par groupe
ind = Y['0']==1
print(X.loc[ind,].mean())
print(X.loc[-ind,].mean())




from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
print(y_train.mean())
print(y_test.mean())# On vérifie que le déséquilibre est respecté dans train et test




SK_logit = LogisticRegression(max_iter =2000, penalty=None)#, fit_intercept=False)
# Problème de pénalité
SK_logit.fit(X_train, y_train)




SK_logit.intercept_.reshape(1,1).shape




SK_logit.intercept_




list(X.columns)




pd.DataFrame(np.concatenate([SK_logit.intercept_.reshape(1,1),
                             SK_logit.coef_],axis=1),
             index = ["coef"],
             columns = ["constante"]+list(X.columns))




pd.DataFrame(np.concatenate([SK_logit.intercept_.reshape(1,1),
                             SK_logit.coef_],axis=1),
             index = ["coef"],
             columns = ["constante"]+list(X.columns)).T




params = SK_logit.coef_


pd.DataFrame(np.exp(params),
             index = ["coef"],
             columns = list(X.columns)).T




import seaborn as sns
from sklearn.metrics import confusion_matrix

conf = confusion_matrix(y_test, SK_logit.predict(X_test))
print(conf)




y_test.shape




(y_test==1).sum()




(y_test==0).sum()




pd.DataFrame(conf, index =['true 0', 'true 1'], columns = ['pred 0', 'pred 1'])




sns.heatmap(conf, annot=True)





conf = confusion_matrix(y_train, SK_logit.predict(X_train))
sns.heatmap(conf, annot=True)




s = 1/4
Y_pred = SK_logit.predict_proba(X_test)[:,1] > s
conf = confusion_matrix(y_test, Y_pred)
sns.heatmap(conf, annot=True)




err = []
prob = SK_logit.predict_proba(X_test)[:,1]
for s in np.arange(1,21)/40:
    Y_pred = SK_logit.predict_proba(X_test)[:,1] > s
    conf = confusion_matrix(y_test, Y_pred)
    err.append(conf[0,1]+10*conf[1,0])




plt.plot(np.arange(1,21)/40, err)



s_opt = np.arange(1,21)[np.argmin(np.array(err))]/40
s_opt



Y_pred = SK_logit.predict_proba(X_test)[:,1] > s_opt
conf = confusion_matrix(y_test, Y_pred)
sns.heatmap(conf, annot=True)




conf[0,1]+10*conf[1,0]




from imblearn.under_sampling import NearMiss
# Choix de la taille du nouveau dataset 
# Sous-Echantillonnage en utilisant la méthode NearMiss 
nearmiss = NearMiss()
X_under_sample, y_under_sample = nearmiss.fit_resample(X, Y)




(y_under_sample==0).sum()



(y_under_sample==1).sum()



y_under_sample.mean()



y_under_sample.shape



SK_logit2 = LogisticRegression(max_iter =2000, penalty=None)

SK_logit2.fit(X_under_sample, y_under_sample)

# Sorties: Coeffs, odds ratio

pd.DataFrame(np.concatenate([SK_logit2.intercept_.reshape(-1,1),
                             SK_logit2.coef_],axis=1),
             index = ["coef"],
             columns = ["constante"]+list(X.columns)).T



params = SK_logit2.coef_


pd.DataFrame(np.exp(params),
             index = ["coef"],
             columns = list(X.columns)).T




conf = confusion_matrix(y_test, SK_logit2.predict(X_test))
sns.heatmap(conf, annot=True)
# erreur pondérée
print("erreur pondérée :",conf[0,1]+10*conf[1,0])




from imblearn.over_sampling import SMOTE
# Choix de la taille du nouveau dataset 
# Sur-Echantillonnage en utilisant la méthode SMOTE
smote = SMOTE()
X_over_sample, y_over_sample = smote.fit_resample(X,Y)
y_over_sample.mean()



SK_logit3 = LogisticRegression(max_iter =2000, penalty=None)#, fit_intercept=False)
# Problème de pénalité
SK_logit3.fit(X_over_sample, y_over_sample)



# Sorties: Coeffs, odds ratio

pd.DataFrame(np.concatenate([SK_logit3.intercept_.reshape(-1,1),
                             SK_logit3.coef_],axis=1),
             index = ["coef"],
             columns = ["constante"]+list(X.columns)).T



params = SK_logit3.coef_


pd.DataFrame(np.exp(params),
             index = ["coef"],
             columns = list(X.columns)).T



conf = confusion_matrix(y_test, SK_logit3.predict(X_test))
sns.heatmap(conf, annot=True)
# erreur pondérée
print("erreur pondérée :",conf[0,1]+10*conf[1,0])



import statsmodels.api as sm
# on ajoute une colonne pour la constante
x_stat = sm.add_constant(X_train)
# on ajuste le modèle
model = sm.Logit(y_train, x_stat)
result = model.fit()
print(result.summary())



from sklearn.metrics import roc_curve, auc
probas = SK_logit.predict_proba(X_test)
print(probas)



probas3 = SK_logit3.predict_proba(X_test)
print(probas3)


fpr0, tpr0, thresholds0 = roc_curve(y_test, probas[:, 1], pos_label=1, drop_intermediate=False)



plt.figure()
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr0, tpr0)
plt.show()



auc_SK = auc(fpr0, tpr0) 
print(auc_SK)



probas2 = SK_logit2.predict_proba(X_test)
fpr2, tpr2, thresholds0 = roc_curve(y_test, probas2[:, 1], pos_label=1, drop_intermediate=False)
probas3 = SK_logit3.predict_proba(X_test)
fpr3, tpr3, thresholds0 = roc_curve(y_test, probas3[:, 1], pos_label=1, drop_intermediate=False)
auc_SK2 = auc(fpr2, tpr2) 
auc_SK3 = auc(fpr3, tpr3) 



fig, ax = plt.subplots(1, 1, figsize=(4,4))
ax.plot([0, 1], [0, 1], 'k--')


ax.plot(fpr0, tpr0, label= 'Init auc=%1.5f' % auc_SK)
ax.plot(fpr2, tpr2, label= 'Under auc=%1.5f' % auc_SK2)
ax.plot(fpr3, tpr3, label= 'Over auc=%1.5f' % auc_SK3)
ax.set_title('Courbe ROC')

ax.set_xlabel("FPR")
ax.set_ylabel("TPR");
ax.legend();





