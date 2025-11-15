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
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.2, random_state = 42, shuffle=True, stratify=y)


RandomForestModel = RandomForestClassifier(random_state=42, n_jobs=-1, class_weight='balanced')
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
    scoring='recall_weighted',      
    n_jobs=-1,               # utiliser tous les cœurs CPU
    verbose=2
)


grid_search.fit(X_train, y_train.values.ravel())

print("Meilleurs hyperparamètres trouvés :")
print(grid_search.best_params_)

best_model = grid_search.best_estimator_

y_proba = best_model.predict_proba(X_test)[:,1]

#Résultats
import sys
sys.path.append("/Users/timothee/Documents/Université Lille/M2/Data Sciences/Projet")
from results_fct import evaluate_and_export_model, compute_CI
results = []
results.append(evaluate_and_export_model(y_test, y_proba, "Rand.Forest.Class", override='Yes'))






















#Intervalles de confiance
import csv
import os

output_dir = "/Users/timothee/Documents/Université Lille/M2/Data Sciences/Projet/Résultats"
os.makedirs(output_dir, exist_ok=True)

output_path = os.path.join(output_dir, "results_ci.csv")

# --- Calcul du CI ---
CI = compute_CI(y_test, y_proba)
model_name = "Random_Forest"

# Vérifie si le fichier existe déjà (pour savoir si on écrit l'en-tête)
file_exists = os.path.isfile(output_path)

with open(output_path, "a", newline="") as f:
    writer = csv.writer(f)
    
    # Si le fichier vient juste d'être créé, écrire l'en-tête
    if not file_exists:
        writer.writerow(["model", "y_true", "ci_low", "ci_high"])
    
    # Ajouter la ligne du modèle
    writer.writerow([model_name, CI[0], CI[1], CI[2]])

print("Fichier enregistré dans :", os.path.abspath(output_path))

