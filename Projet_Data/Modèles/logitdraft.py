#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 28 12:58:40 2025

@author: timothee
"""

from ucimlrepo import fetch_ucirepo 
  
# fetch dataset 
breast_cancer_wisconsin_diagnostic = fetch_ucirepo(id=17) 
  
# data (as pandas dataframes) 
X = breast_cancer_wisconsin_diagnostic.data.features
y = breast_cancer_wisconsin_diagnostic.data.targets 
y = y.replace({"M": 1, "B": 0}).astype(float)
# metadata 
print(breast_cancer_wisconsin_diagnostic.metadata) 
  
# variable information 
print(breast_cancer_wisconsin_diagnostic.variables) 

import numpy as np
import sklearn 
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns 
from sklearn import metrics
from imblearn.pipeline import Pipeline
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve, auc
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.2, random_state = 42, shuffle=True, stratify=y)






param_grid = [
    {'penalty': ['l2'], 'C': [0.01, 0.1, 1, 10, 100],
     'solver': ['lbfgs', 'newton-cg', 'newton-cholesky', 'sag']},

    {'penalty': ['l1'], 'C': [0.01, 0.1, 1, 10, 100],
     'solver': ['liblinear', 'saga']},

    {'penalty': ['elasticnet'], 'C': [0.01, 0.1, 1, 10, 100],
     'solver': ['saga'], 'l1_ratio': [0, 0.5, 1]},

    {'penalty': [None], 'solver': ['lbfgs', 'newton-cg', 'newton-cholesky', 'saga']}
]

baseline_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model', LogisticRegression(random_state=42, max_iter=9000))
])

baseline_param_grid = []
for params in param_grid:
    pipeline_params = {}
    for key, value in params.items():
        pipeline_params[f'model__{key}'] = value
    baseline_param_grid.append(pipeline_params)
    
def plot_roc_curve(y_true, y_proba, title='Courbe ROC'):
    
    fpr, tpr, thresholds = roc_curve(y_true, y_proba, drop_intermediate=False)
    roc_auc = roc_auc_score(y_true, y_proba)
    
    # Indice de Youden
    youden_index = tpr - fpr
    optimal_idx = np.argmax(youden_index)
    optimal_threshold = thresholds[optimal_idx]
    
    # Plot ROC
    plt.figure(figsize=(8, 8))
    plt.plot(fpr, tpr, color='blue', lw=3, label=f'AUC = {roc_auc:.3f}')
    plt.plot([0, 1], [0, 1], color='red', lw=2, linestyle='--', label='Aléatoire')
    plt.scatter(fpr[optimal_idx], tpr[optimal_idx], color='green', s=100, 
                label=f'Seuil optimal = {optimal_threshold:.2f}')
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('Taux de Faux Positifs (FPR)')
    plt.ylabel('Taux de Vrais Positifs (TPR)')
    plt.title(title)
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    print(f"AUC: {roc_auc:.3f}")
    print(f"Seuil optimal: {optimal_threshold:.3f}")
    
    return optimal_threshold, roc_auc

# GridSearch baseline
baseline_grid_search = GridSearchCV(
    estimator=baseline_pipeline,
    param_grid=baseline_param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
    verbose=1
)

baseline_grid_search.fit(X_train, y_train.values.ravel())
print("Meilleurs hyperparamètres :", baseline_grid_search.best_params_)
print(f"Meilleure accuracy moyenne (CV) : {baseline_grid_search.best_score_:.4f}")

baseline_model = baseline_grid_search.best_estimator_
baseline_y_proba = baseline_model.predict_proba(X_test)[:, 1]

optimal_threshold_baseline, roc_auc_baseline = plot_roc_curve(y_test, baseline_y_proba)

y_pred_thresh_baseline = (baseline_y_proba >= optimal_threshold_baseline).astype(int)
classification_report_baseline = classification_report(y_test, y_pred_thresh_baseline, target_names=['Bégnin', 'Malin'])
cm_baseline = confusion_matrix(y_test, y_pred_thresh_baseline)
acc_baseline = metrics.accuracy_score(y_test, y_pred_thresh_baseline)







#Stratégie 2
from imblearn.under_sampling import NearMiss
NM_Pipeline = Pipeline([
    ('sampling', NearMiss(version=1, n_jobs=-1)),
    ('scaler', StandardScaler()),
    ('model', LogisticRegression(random_state=42, max_iter=9000))
])

NM_param_grid = []
for params in param_grid:
    pipeline_params = {}
    for key, value in params.items():
        pipeline_params[f'model__{key}'] = value
    NM_param_grid.append(pipeline_params)
    
NM_grid_search = GridSearchCV(
    estimator=NM_Pipeline,
    param_grid=NM_param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
    verbose=1
)

NM_grid_search.fit(X_train, y_train.values.ravel())
print("Meilleurs hyperparamètres :", NM_grid_search.best_params_)
print(f"Meilleure accuracy moyenne (CV) : {NM_grid_search.best_score_:.4f}")

NM_model = NM_grid_search.best_estimator_
NM_y_proba = NM_model.predict_proba(X_test)[:, 1]


optimal_threshold_NM, roc_auc_NM = plot_roc_curve(y_test, NM_y_proba)

y_pred_thresh_NM = (NM_y_proba >= optimal_threshold_NM).astype(int)
classififcation_report_NM = classification_report(y_test, y_pred_thresh_NM, target_names=['Bégnin', 'Malin'])
cm_NM = confusion_matrix(y_test, y_pred_thresh_NM)
acc_NM = metrics.accuracy_score(y_test, y_pred_thresh_NM)



#Stratégie3
from imblearn.over_sampling import SMOTE
smote_pipeline = Pipeline([
    ('sampling', SMOTE(random_state=42, n_jobs=-1)),
    ('scaler', StandardScaler()),
    ('model', LogisticRegression(random_state=42, max_iter=9000))
])

smote_param_grid = []
for params in param_grid:
    for k in [2, 5, 8, 10]:
        pipeline_params = {}
        for key, value in params.items():
            pipeline_params[f'model__{key}'] = value
        pipeline_params['sampling__k_neighbors'] = [k]
        smote_param_grid.append(pipeline_params)
        
smote_grid_search = GridSearchCV(
    estimator=smote_pipeline,
    param_grid=smote_param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
    verbose=1
)

smote_grid_search.fit(X_train, y_train.values.ravel())
print("Meilleurs hyperparamètres :", smote_grid_search.best_params_)
print(f"Meilleure accuracy moyenne (CV) : {smote_grid_search.best_score_:.4f}")
smote_model = smote_grid_search.best_estimator_
smote_y_proba = smote_model.predict_proba(X_test)[:, 1]


optimal_threshold_smote, roc_auc_smote = plot_roc_curve(y_test, smote_y_proba)

y_pred_thresh_smote = (smote_y_proba >= optimal_threshold_smote).astype(int)
classification_report_smote = classification_report(y_test, y_pred_thresh_smote, target_names=['Bégnin', 'Malin'])
cm_smote = confusion_matrix(y_test, y_pred_thresh_smote)
acc_smote = metrics.accuracy_score(y_test, y_pred_thresh_smote)







