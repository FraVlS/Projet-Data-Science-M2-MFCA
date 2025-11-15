#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 26 13:28:21 2025

@author: timothee
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 26 10:19:36 2025

@author: timothee
"""

from ucimlrepo import fetch_ucirepo 
  

breast_cancer_wisconsin_diagnostic = fetch_ucirepo(id=17) 
  

X = breast_cancer_wisconsin_diagnostic.data.features
y = breast_cancer_wisconsin_diagnostic.data.targets 
y = y.replace({"M": 1, "B": 0}).astype(float)  
# metadata 
print(breast_cancer_wisconsin_diagnostic.metadata) 
  

print(breast_cancer_wisconsin_diagnostic.variables) 

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns 
from sklearn import metrics
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from imblearn.under_sampling import NearMiss
# Matrice de corrélation
corr = X.corr(method="spearman")
corr_mask = np.abs(corr) > 0.6  
corr_highlighted = corr.copy()  
corr_highlighted[~corr_mask] = np.nan  
plt.figure(figsize=(20, 10))
ax = sns.heatmap(corr_highlighted, annot=True, cmap='coolwarm', fmt='.2f', annot_kws={"size": 14},
                                vmin=-1, vmax=1)  # Set the color scale to range from -1 to 1
plt.title("Correlation Matrix", fontsize=16)
plt.xticks(rotation=45, fontsize=12)
plt.yticks(rotation=0, fontsize=12)
ax.set_xticklabels(corr.columns, fontsize=12)
ax.set_yticklabels(corr.index, fontsize=12)
plt.show()


def split_data(X, y):
    X_train_full, X_test, y_train_full, y_test = train_test_split(X, y, test_size=0.20, random_state=42, shuffle=True, stratify=y)
    X_train, X_valid, y_train, y_valid = train_test_split(X_train_full, y_train_full, test_size=0.125, random_state= 42, shuffle=True, stratify=y_train_full)
    X_index = X_test.index.to_frame(index=True)
    nm = NearMiss(version=1)
    X_train, y_train = nm.fit_resample(X_train, y_train)
    return X_test, y_test, X_train, y_train, X_valid, y_valid, X_index


X_test, y_test, X_train, y_train, X_valid, y_valid, X_index = split_data(X, y)

import tensorflow as tf
from tensorflow import keras
import sys
import os


root_logdir = os.path.join(os.curdir, "my_logs")

def get_run_logdir():
    import time
    run_id = time.strftime("run_%Y_%m_%d-%H_%M_%S")
    return os.path.join(root_logdir, run_id)


run_logdir = get_run_logdir()
tensorboard_cb = tf.keras.callbacks.TensorBoard(run_logdir)
def build_logistic_regression(X, learning_rate=0.001):
    """
    Construit un modèle de régression logistique avec Keras.
    X : jeu de données d'entraînement (pour déterminer le nombre de features)
    learning_rate : taux d'apprentissage de l'optimiseur Adam
    """
    model = Sequential()
    
    # Couche unique : régression logistique
    model.add(Dense(1, input_dim=X.shape[1], activation='sigmoid'))
    
    
    optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=learning_rate)
    
   
    model.compile(
        loss='binary_crossentropy',   # perte logistique
        optimizer=optimizer,
        metrics=[tf.keras.metrics.Recall(name='recall')]
    )
    
    model.summary()
    return model
def train_model(model, X_train, X_valid, y_train, y_valid):
   
    checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
        "Logit_model_NM.h5",  
        save_best_only=True
    )    
    
    early_stopping_cb = tf.keras.callbacks.EarlyStopping(
        patience=20,
        restore_best_weights=True
    )
    
  
    
    

    history = model.fit(
        X_train, y_train, 
        epochs=200,
        validation_data=(X_valid, y_valid),
        callbacks=[checkpoint_cb, early_stopping_cb, tensorboard_cb],
        verbose=1  
    )
    
   
    learning_curves = pd.DataFrame(history.history).plot(figsize=(8, 5))
    plt.title("Courbes d'apprentissage")
    plt.grid(True)
    
    return history, learning_curves

# ML_Model


from sklearn.preprocessing import StandardScaler

def preprocess_data(X_train, X_valid, X_test):
    """Standardise les données pour améliorer la convergence"""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_valid_scaled = scaler.transform(X_valid)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_valid_scaled, X_test_scaled, scaler


print("Préparation des données...")


X_train_np = X_train.to_numpy().astype(np.float32)
X_valid_np = X_valid.to_numpy().astype(np.float32)
X_test_np  = X_test.to_numpy().astype(np.float32)


y_train_np = y_train.to_numpy().astype(np.float32).reshape(-1,1)
y_valid_np = y_valid.to_numpy().astype(np.float32).reshape(-1,1)
y_test_np  = y_test.to_numpy().astype(np.float32).reshape(-1,1)


X_train_scaled, X_valid_scaled, X_test_scaled, scaler = preprocess_data(
    X_train_np, X_valid_np, X_test_np
)



model = build_logistic_regression(X_train_scaled)

history, learning_curves = train_model(model, X_train_scaled, X_valid_scaled, y_train_np, y_valid_np)


eval_results = model.evaluate(X_test_scaled, y_test_np, verbose=0)



save_dir = "/Users/timothee/Documents/Université Lille/M2/Data Sciences/Projet"
os.makedirs(save_dir, exist_ok=True)
model_path = os.path.join(save_dir, "Logit_model_NM.h5")
model.save(model_path)
print("\nModèle sauvegardé sous 'Logit_model_NM.h5")
model = keras.models.load_model("/Users/timothee/Documents/Université Lille/M2/Data Sciences/Projet/Logit_model_NM.h5")
y_proba = model.predict(X_test_scaled)


#Résultats
sys.path.append("/Users/timothee/Documents/Université Lille/M2/Data Sciences/Projet")
from results_fct import evaluate_and_export_model
results = []
results.append(evaluate_and_export_model(y_test, y_proba, "Reglog.early-stop_NM",override="Yes"))


