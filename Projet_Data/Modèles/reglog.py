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











#Résultats
import sys
sys.path.append("/Users/timothee/Documents/Université Lille/M2/Data Sciences/Projet")
from results_fct import evaluate_and_export_model
results = []
results.append(evaluate_and_export_model(y_test, baseline_y_proba, "Logit_Baseline"))
results.append(evaluate_and_export_model(y_test, NM_y_proba, "Logit_NearMiss"))
results.append(evaluate_and_export_model(y_test, smote_y_proba, "Logit_SMOTE"))