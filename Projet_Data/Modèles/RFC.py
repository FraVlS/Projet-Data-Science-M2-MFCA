#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 26 18:24:54 2025

@author: timothee
"""



from ucimlrepo import fetch_ucirepo 
  

breast_cancer_wisconsin_diagnostic = fetch_ucirepo(id=17) 
  

X = breast_cancer_wisconsin_diagnostic.data.features
y = breast_cancer_wisconsin_diagnostic.data.targets 
y = y.replace({"M": 1, "B": 0}).astype(float)  

print(breast_cancer_wisconsin_diagnostic.metadata) 
  

print(breast_cancer_wisconsin_diagnostic.variables) 

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.2, random_state = 42, shuffle=True, stratify=y)

def preprocess_data(X_train, X_test):
    """Standardise les données pour améliorer la convergence"""
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train, X_test, scaler

RandomForestModel = RandomForestClassifier(random_state=42, n_jobs=-1)
param_grid = {
    'n_estimators': [100, 300, 500],
    'criterion': ['gini', 'entropy', 'log_loss'],
    'max_depth': [None, 5, 10, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2']
}
grid_search = GridSearchCV(
    estimator=RandomForestModel,
    param_grid=param_grid,
    cv=5,                    
    scoring='accuracy',      
    n_jobs=-1,               # utiliser tous les cœurs CPU
    verbose=2
)
X_train, X_test, scaler = preprocess_data(X_train, X_test)

grid_search.fit(X_train, y_train.values.ravel())

print("Meilleurs hyperparamètres trouvés :")
print(grid_search.best_params_)

print("\n Meilleure précision en validation croisée :")
print(grid_search.best_score_)

best_model = grid_search.best_estimator_

y_pred = best_model.predict(X_test)

print(classification_report(y_test, y_pred, target_names=["Benin (0)", "Malin (1)"]))



cm = confusion_matrix(y_test, y_pred)

#Courbe ROC
import numpy as np
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve, auc
y_pred_proba = best_model.predict_proba(X_test)[:,1]
import matplotlib.pyplot as plt

fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
roc_auc = roc_auc_score(y_test, y_pred_proba)

# Point optimal (indice de Youden)
youden_index = tpr - fpr
optimal_idx = np.argmax(youden_index)
optimal_threshold = thresholds[optimal_idx]


plt.figure(figsize=(8, 8))


plt.plot(fpr, tpr, color='blue', lw=3, 
         label=f'AUC = {roc_auc:.3f}')


plt.plot([0, 1], [0, 1], color='red', lw=2, linestyle='--', 
         label='Aléatoire')

# Point optimal
plt.scatter(fpr[optimal_idx], tpr[optimal_idx], color='green', s=100, 
           label=f'Seuil optimal = {optimal_threshold:.2f}')


plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('Taux de Faux Positifs (FPR)')
plt.ylabel('Taux de Vrais Positifs (TPR)')
plt.title('Courbe ROC')
plt.legend(loc='lower right')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print(f"AUC: {roc_auc:.3f}")
print(f"Seuil optimal: {optimal_threshold:.3f}")