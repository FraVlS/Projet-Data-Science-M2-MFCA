#########################
# Regression logistique #
#########################
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

data = pd.read_csv("heart_cleveland.csv")
X = data.drop('condition', axis = 1)
Y = data.condition

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
print(y_train.mean())
print(y_test.mean())# 

SK_logit = LogisticRegression(max_iter =2000, penalty=None)
# Attention a la pénalité
SK_logit.fit(X_train, y_train)

SK_logit.intercept_.reshape(1,1).shape

# Estimation des parametres avec l'intercept
pd.DataFrame(np.concatenate([SK_logit.intercept_.reshape(1,1),
                              SK_logit.coef_],axis=1),index = ["coef"],columns = ["constante"]+list(X.columns))

# Estimation des odds ratios
params = SK_logit.coef_
pd.DataFrame(np.exp(params),
             index = ["coef"],
             columns = list(X.columns)).T

from sklearn.metrics import confusion_matrix

conf = confusion_matrix(y_test, SK_logit.predict(X_test))
pd.DataFrame(conf, index =['true 0', 'true 1'], columns = ['pred 0', 'pred 1'])
plt.figure()
sns.heatmap(conf, annot=True)
plt.show()

# Pour obtenir les IC et tests
import statsmodels.api as sm
# on ajoute une colonne pour la constante
x_stat = sm.add_constant(X_train)
# on ajuste le modèle
model = sm.Logit(y_train, x_stat)
result = model.fit()
print(result.summary())

# Courbe ROC
prob = SK_logit.predict_proba(X_test)[:, 1]
thresholds = np.arange(1, 4001) / 4000.0
err = []
TP = []
VN = []

for s in thresholds:
    Y_pred = prob > s
    # ensure a 2x2 confusion matrix in the order [0,1]
    conf = confusion_matrix(y_test, Y_pred,labels=[0, 1])
    # conf layout: [[TN, FP],
    #               [FN, TP]]
    err.append(conf[0, 1] + conf[1, 0])
    TP.append(conf[1, 1])
    VN.append(conf[0, 0])

err = np.array(err, dtype=float)/y_test.size
TP = np.array(TP, dtype=float)/sum(y_test)
FP = 1-np.array(VN, dtype=float)/(y_test.size-sum(y_test))

plt.figure()
plt.plot(FP[::-1], TP[::-1], 'r.')
plt.step(FP[::-1], TP[::-1],where='pre',label='Courbe ROC')
plt.plot([0, 1], [0, 1], 'k--',label='Modele nul')
plt.xlabel('1 - Specificite')
plt.ylabel('Sensibilite')
plt.legend(loc='lower right')
plt.grid(alpha=0.3)
plt.show()

# Meme chose avec le package AUC
from sklearn.metrics import roc_curve, auc    

fpr0, tpr0, thresholds0 = roc_curve(y_test, prob, drop_intermediate=False)

plt.figure()
plt.plot([0, 1], [0, 1], 'k--',label='Modele nul')
plt.plot(fpr0, tpr0)
plt.grid(alpha=0.3)
plt.xlabel('1 - Specificite')
plt.ylabel('Sensibilite')
plt.legend(loc='lower right')
plt.show()