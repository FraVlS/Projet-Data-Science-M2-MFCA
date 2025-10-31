#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 30 10:44:52 2025

@author: timothee
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix, classification_report
import numpy as np
def evaluate_and_export_model(y_true, y_proba, model_name, export_dir="/Users/timothee/Documents/Université Lille/M2/Data Sciences/Projet/Résultats", override = "No", w=0.8):
    if override.lower() not in ["yes", "no"]:
        raise ValueError("override doit être 'yes' ou 'no'")
    fpr, tpr, thresholds = roc_curve(y_true, y_proba, drop_intermediate=False)
    roc_auc = roc_auc_score(y_true, y_proba) 
    if override.lower() == "yes":
        youden_index = w*tpr - (1-w)*fpr
    else:
        youden_index = tpr - fpr
        

    optimal_idx = np.argmax(youden_index)
    optimal_threshold = thresholds[optimal_idx]
    y_pred = (y_proba >= optimal_threshold).astype(int)

    report = classification_report(y_true, y_pred, target_names=['Bénin', 'Malin'], output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    report_df.to_csv(f"{export_dir}/{model_name}_classification_report.csv")

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=['Prédit: Bénin', 'Prédit: Malin'],
                yticklabels=['Réel: Bénin', 'Réel: Malin'])
    plt.title(f'Matrice de Confusion - {model_name}')
    plt.tight_layout()
    plt.savefig(f"{export_dir}/{model_name}_confusion_matrix.png", dpi=300)
    plt.close()

    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, color='blue', lw=2, label=f'AUC = {roc_auc:.3f}')
    plt.plot([0, 1], [0, 1], color='red', lw=2, linestyle='--')
    plt.scatter(fpr[optimal_idx], tpr[optimal_idx], color='green', s=100,
                label=f'Seuil optimal = {optimal_threshold:.2f}')
    plt.xlabel('Faux positifs (FPR)')
    plt.ylabel('Vrais positifs (TPR)')
    plt.title(f'Courbe ROC - {model_name}')
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{export_dir}/{model_name}_ROC_curve.png", dpi=300)
    plt.close()

    print(f"\n==== {model_name} ====")
    print(f"AUC: {roc_auc:.3f}")
    print(f"Seuil optimal: {optimal_threshold:.3f}")
    print(f"Accuracy: {report['accuracy']:.3f}")
    print("Classification report enregistré sous :", f"{export_dir}/{model_name}_classification_report.csv")

    return {
        "model": model_name,
        "AUC": roc_auc,
        "Optimal_Threshold": optimal_threshold,
        "Accuracy": report["accuracy"]
    }