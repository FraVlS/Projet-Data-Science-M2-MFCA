
# Breast Cancer Wisconsin Dataset - Préliminaire
# Study on the Wisconsin Diagnostic Breast Cancer (WDBC) dataset
# https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

# 1. Importation et préparation des données
# Read data without header
df = pd.read_csv('wdbc.data', header=None)

# Create column names
feature_names = ['ID', 'Diagnosis']
# Add the 30 feature names
feat = ['radius', 'texture', 'perimeter', 'area', 'smoothness', 'compactness',
        'concavity', 'concave_points', 'symmetry', 'fractal_dimension']
features = []
for f in feat:
    features.append(f + '_mean')
    features.append(f + '_se')
    features.append(f + '_worst')

# Concatenate feature names
feature_names.extend(features)
df.columns = feature_names

# Display basic information
print("="*80)
print("DESCRIPTION DE LA BASE DE DONNÉES")
print("="*80)
print("\nNombre d'individus:", df.shape[0])
print("Nombre de variables:", df.shape[1])
print("\nVariables:")
print(df.columns.tolist()[:5], "...", df.columns.tolist()[-5:])

# 2. Description de la variable cible (Diagnosis)
print("\n" + "="*80)
print("ÉTUDE DE LA VARIABLE CIBLE: DIAGNOSIS")
print("="*80)

print("\nRépartition de la classe:")
print(df['Diagnosis'].value_counts())
print("\nPourcentages:")
print(df['Diagnosis'].value_counts(normalize=True) * 100)

# Check class balance
benign_count = (df['Diagnosis'] == 'B').sum()
malignant_count = (df['Diagnosis'] == 'M').sum()
total = len(df)
imbalance_ratio = benign_count / malignant_count

print(f"\nRatio d'équilibre (B/M): {imbalance_ratio:.3f}")
print(f"Benign: {benign_count} ({benign_count/total*100:.1f}%)")
print(f"Malignant: {malignant_count} ({malignant_count/total*100:.1f}%)")

if abs(imbalance_ratio - 1) > 0.3:
    print("\n⚠️  Les classes sont déséquilibrées.")
    print("   Considération: Il faudra potentiellement mettre en œuvre")
    print("   un algorithme de rééquilibrage (undersampling/oversampling).")
else:
    print("\n✓  Les classes sont relativement équilibrées.")
    
# Visualisation de la répartition
plt.figure(figsize=(8, 6))
df['Diagnosis'].value_counts().plot(kind='bar', color=['green', 'red'], rot=0)
plt.title('Répartition des classes: Benign vs Malignant')
plt.xlabel('Diagnosis')
plt.ylabel('Nombre d\'individus')
plt.xticks([0, 1], ['Benign (B)', 'Malignant (M)'])
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('class_distribution.png')
plt.show()

# 3. Statistiques descriptives des variables explicatives
print("\n" + "="*80)
print("STATISTIQUES DESCRIPTIVES UNIVARIÉES")
print("="*80)

# Separate features
X = df.drop(['ID', 'Diagnosis'], axis=1)

print("\nStatistiques descriptives générales:")
print(X.describe())

# Statistiques descriptives par classe
print("\nStatistiques descriptives par classe:")
print("\n--- BENIGN ---")
X_benign = X[df['Diagnosis'] == 'B']
print(X_benign.describe())

print("\n--- MALIGNANT ---")
X_malignant = X[df['Diagnosis'] == 'M']
print(X_malignant.describe())

# Difference in means between classes
print("\n" + "="*80)
print("DIFFÉRENCES DE MOYENNES ENTRE CLASSES")
print("="*80)
mean_diff = X_benign.mean() - X_malignant.mean()
print("\nTop 10 variables avec les plus grandes différences de moyennes:")
print(mean_diff.abs().sort_values(ascending=False).head(10))

# Selected features for bivariate analysis
selected_features = mean_diff.abs().sort_values(ascending=False).head(6).index.tolist()
print(f"\nVariables sélectionnées pour l'analyse bivariée: {selected_features}")

# 4. Analyse bivariée - Corrélations
print("\n" + "="*80)
print("ANALYSE BIVARIÉE - MATRICE DE CORRÉLATION")
print("="*80)

correlation_matrix = X.corr()
print("\nMatrice de corrélation des variables explicatives:")
print(correlation_matrix)

# Visualisation de la matrice de corrélation
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm', center=0, 
            square=True, linewidths=0.5, cbar_kws={"shrink": 0.8})
plt.title('Matrice de corrélation des variables explicatives')
plt.tight_layout()
plt.savefig('correlation_matrix.png')
plt.show()

# Visualisation zoomée sur les variables les plus discriminantes
plt.figure(figsize=(12, 10))
sns.heatmap(X[selected_features].corr(), annot=True, cmap='coolwarm', 
            center=0, square=True, linewidths=0.5, fmt='.2f')
plt.title('Matrice de corrélation - Variables les plus discriminantes')
plt.tight_layout()
plt.savefig('correlation_selected.png')
plt.show()

# 5. Analyse bivariée - Relation variables / classe
print("\n" + "="*80)
print("ANALYSE BIVARIÉE - RÉPARTITION DES VARIABLES PAR CLASSE")
print("="*80)

# Créer des boîtes à moustaches pour quelques variables importantes
n_vars = min(6, len(selected_features))
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()

for i, var in enumerate(selected_features[:n_vars]):
    data_to_plot = [X_benign[var], X_malignant[var]]
    bp = axes[i].boxplot(data_to_plot, labels=['Benign', 'Malignant'], 
                        patch_artist=True)
    
    # Color boxes
    colors = ['lightgreen', 'salmon']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
    
    axes[i].set_title(f'{var}')
    axes[i].grid(axis='y', alpha=0.3)

plt.suptitle('Distribution des variables importantes par classe', fontsize=14)
plt.tight_layout()
plt.savefig('boxplots_by_class.png')
plt.show()

# 6. Histogrammes des variables principales
print("\nVisualisation des distributions des variables principales...")

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()

for i, var in enumerate(selected_features[:n_vars]):
    axes[i].hist(X_benign[var], alpha=0.6, label='Benign', color='green', bins=20)
    axes[i].hist(X_malignant[var], alpha=0.6, label='Malignant', color='red', bins=20)
    axes[i].set_title(f'{var}')
    axes[i].legend()
    axes[i].grid(axis='y', alpha=0.3)

plt.suptitle('Histogrammes des variables principales par classe', fontsize=14)
plt.tight_layout()
plt.savefig('histograms_by_class.png')
plt.show()

# 7. Scatter plots pour les paires de variables les plus discriminantes
if len(selected_features) >= 2:
    print("\nGraphes de dispersion - Relations entre variables discriminantes...")
    
    # Plot scatter pour les 3 premières variables les plus discriminantes
    fig = plt.figure(figsize=(15, 5))
    
    var_pairs = [
        (selected_features[0], selected_features[1]),
        (selected_features[0], selected_features[2]),
        (selected_features[1], selected_features[2])
    ]
    
    for idx, (var1, var2) in enumerate(var_pairs):
        ax = plt.subplot(1, 3, idx + 1)
        benign_indices = df['Diagnosis'] == 'B'
        malignant_indices = df['Diagnosis'] == 'M'
        
        ax.scatter(df[var1][benign_indices], df[var2][benign_indices], 
                   alpha=0.6, label='Benign', color='green', s=30)
        ax.scatter(df[var1][malignant_indices], df[var2][malignant_indices], 
                   alpha=0.6, label='Malignant', color='red', s=30)
        
        ax.set_xlabel(var1)
        ax.set_ylabel(var2)
        ax.legend()
        ax.grid(alpha=0.3)
    
    plt.suptitle('Graphes de dispersion des variables les plus discriminantes', fontsize=14)
    plt.tight_layout()
    plt.savefig('scatter_plots.png')
    plt.show()

# 8. Résumé et conclusions
print("\n" + "="*80)
print("RÉSUMÉ DE L'ÉTUDE PRÉLIMINAIRE")
print("="*80)
print(f"""
1. DONNÉES:
   - Nombre total d'individus: {df.shape[0]}
   - Nombre de variables explicatives: {X.shape[1]}
   - Classes: Benign (B) et Malignant (M)

2. ÉQUILIBRE DES CLASSES:
   - Benign: {benign_count} ({benign_count/total*100:.1f}%)
   - Malignant: {malignant_count} ({malignant_count/total*100:.1f}%)
   - Ratio B/M: {imbalance_ratio:.3f}

3. VARIABLES LES PLUS DISCRIMINANTES:
""")
for i, var in enumerate(selected_features[:5], 1):
    diff = abs(X_benign[var].mean() - X_malignant[var].mean())
    print(f"   {i}. {var}: différence = {diff:.4f}")

print("""
4. RECOMMANDATIONS:
""")
if imbalance_ratio > 1.5 or imbalance_ratio < 0.67:
    print("   ⚠️  Les classes sont déséquilibrées.")
    print("      → Nécessité potentielle de techniques de rééchantillonnage")
    print("      → Stratification recommandée lors de la division train/test")
else:
    print("   ✓ Les classes sont relativement équilibrées.")
    print("      → Pas de rééquilibrage strictement nécessaire")
    
print("""
   → Préparation des données standard recommandée (centrage/réduction)
   → Division train/test avec stratification sur la variable Diagnosis
   → Méthodes à tester: LDA, QDA, régression logistique, KNN, forêts aléatoires

""")

# 9. Préparation pour les étapes suivantes
# Standardize features
scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

# Encode target variable
y_encoded = df['Diagnosis'].map({'B': 0, 'M': 1})

print("="*80)
print("Préparation des données terminée.")
print("="*80)
print(f"\nDonnées normalisées (X_scaled): shape {X_scaled.shape}")
print(f"Variables cibles encodées (y_encoded): {y_encoded.value_counts().to_dict()}")

# Save prepared data for next steps
# X_scaled et y_encoded sont prêts pour l'apprentissage

print("\n✓ Étude préliminaire terminée avec succès!")

###############################################################################
# ALGORITHMES DE CLASSIFICATION
###############################################################################

print("\n" + "="*80)
print("ALGORITHMES DE CLASSIFICATION")
print("="*80)

# Imports supplémentaires pour les algorithmes
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.metrics import roc_curve, auc, roc_auc_score

# Séparation train/test (stratifié pour maintenir la distribution des classes)
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_encoded, test_size=0.3, random_state=42, stratify=y_encoded
)

print(f"\nDivision des données:")
print(f"  - Train: {X_train.shape[0]} échantillons")
print(f"  - Test: {X_test.shape[0]} échantillons")
print(f"\nDistribution des classes dans le train:")
print(f"  - Benign (0): {(y_train == 0).sum()} ({(y_train == 0).mean()*100:.1f}%)")
print(f"  - Malignant (1): {(y_train == 1).sum()} ({(y_train == 1).mean()*100:.1f}%)")
print(f"\nDistribution des classes dans le test:")
print(f"  - Benign (0): {(y_test == 0).sum()} ({(y_test == 0).mean()*100:.1f}%)")
print(f"  - Malignant (1): {(y_test == 1).sum()} ({(y_test == 1).mean()*100:.1f}%)")

# Fonction pour afficher les résultats
def evaluate_model(model, name, X_train, X_test, y_train, y_test):
    """Évalue un modèle et retourne les métriques"""
    # Prédictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    # Scores
    train_score = accuracy_score(y_train, y_train_pred)
    test_score = accuracy_score(y_test, y_test_pred)
    
    # Matrice de confusion
    conf_matrix = confusion_matrix(y_test, y_test_pred)
    
    print(f"\n{'='*80}")
    print(f"{name}")
    print(f"{'='*80}")
    print(f"\nPrécision sur train: {train_score:.4f}")
    print(f"Précision sur test: {test_score:.4f}")
    
    print(f"\nMatrice de confusion (test):")
    print(conf_matrix)
    print(f"\n         Prédit")
    print(f"         0     1")
    print(f"Réel 0  {conf_matrix[0,0]:4d}  {conf_matrix[0,1]:4d}")
    print(f"      1  {conf_matrix[1,0]:4d}  {conf_matrix[1,1]:4d}")
    
    # Rapport de classification
    print(f"\nRapport de classification:")
    print(classification_report(y_test, y_test_pred, target_names=['Benign', 'Malignant']))
    
    return {
        'name': name,
        'train_score': train_score,
        'test_score': test_score,
        'confusion_matrix': conf_matrix,
        'model': model
    }

# Dictionnaire pour stocker tous les résultats
results = {}

###############################################################################
# 1. LINEAR DISCRIMINANT ANALYSIS (LDA)
###############################################################################

print("\n" + "="*80)
print("1. LINEAR DISCRIMINANT ANALYSIS (LDA)")
print("="*80)

print("\nJustification de LDA:")
print("  - LDA suppose que les deux classes partagent la même matrice de covariance")
print("  - Hypothèse de normalité multivariée")
print("  - Utilise des hyperplans linéaires pour séparer les classes")
print("  - Rapide et simple, bon point de départ pour problèmes binaires")

lda = LinearDiscriminantAnalysis(solver='svd')
lda.fit(X_train, y_train)

results['LDA'] = evaluate_model(lda, "LDA - Linear Discriminant Analysis", 
                                X_train, X_test, y_train, y_test)

# Affichage des moyennes par classe
print(f"\nMoyennes estimées par classe:")
print("Benign (classe 0):")
print(lda.means_[0, :5], "... (premières 5 variables)")
print("Malignant (classe 1):")
print(lda.means_[1, :5], "... (premières 5 variables)")
print(f"\nProbabilités a priori: {lda.priors_}")

###############################################################################
# 2. QUADRATIC DISCRIMINANT ANALYSIS (QDA)
###############################################################################

print("\n" + "="*80)
print("2. QUADRATIC DISCRIMINANT ANALYSIS (QDA)")
print("="*80)

print("\nJustification de QDA:")
print("  - QDA suppose des matrices de covariance différentes pour chaque classe")
print("  - Plus flexible que LDA (frontières quadratiques)")
print("  - Peut capturer des relations non-linéaires")
print("  - Nécessite plus de données que LDA (estime plus de paramètres)")

qda = QuadraticDiscriminantAnalysis()
qda.fit(X_train, y_train)

results['QDA'] = evaluate_model(qda, "QDA - Quadratic Discriminant Analysis",
                                X_train, X_test, y_train, y_test)

print(f"\nProbabilités a priori: {qda.priors_}")

###############################################################################
# 3. RÉGRESSION LOGISTIQUE
###############################################################################

print("\n" + "="*80)
print("3. RÉGRESSION LOGISTIQUE")
print("="*80)

print("\nJustification de la régression logistique:")
print("  - Modèle probabiliste: fournit P(Y=1|X)")
print("  - Interprétable: coefficients indiquent l'impact des variables")
print("  - Pas d'hypothèse de normalité")
print("  - Fonctionne bien pour classification binaire")
print("  - Peut gérer les interactions et non-linéarités via transformations")

logit = LogisticRegression(max_iter=2000, random_state=42)
logit.fit(X_train, y_train)

results['Logistic Regression'] = evaluate_model(
    logit, "Logistic Regression", X_train, X_test, y_train, y_test
)

# Afficher les coefficients les plus importants
coef_df = pd.DataFrame({
    'Variable': X_train.columns,
    'Coefficient': logit.coef_[0]
})
coef_df['Abs_Coefficient'] = coef_df['Coefficient'].abs()
top_coef = coef_df.sort_values('Abs_Coefficient', ascending=False).head(10)

print(f"\nTop 10 variables les plus importantes (par valeur absolue du coefficient):")
print(top_coef[['Variable', 'Coefficient']].to_string(index=False))

# Odds ratios
print(f"\nTop 10 odds ratios (exp(coef)):")
coef_df['Odds_Ratio'] = np.exp(coef_df['Coefficient'])
top_odds = coef_df.sort_values('Abs_Coefficient', ascending=False).head(10)
print(top_odds[['Variable', 'Odds_Ratio', 'Coefficient']].to_string(index=False))

###############################################################################
# 4. K-NEAREST NEIGHBORS (KNN)
###############################################################################

print("\n" + "="*80)
print("4. K-NEAREST NEIGHBORS (KNN)")
print("="*80)

print("\nJustification de KNN:")
print("  - Algorithme non-paramétrique (aucune hypothèse sur la distribution)")
print("  - Se base sur la similarité locale")
print("  - Peut capturer des frontières complexes")
print("  - Sensible à la distance choisie (ici Euclidean avec données standardisées)")
print("\nCalibration des hyperparamètres:")
print("  - Hyperparamètre principal: k (nombre de voisins)")
print("  - Trop petit (k=1): sur-apprentissage, sensible au bruit")
print("  - Trop grand: sous-apprentissage, perd la flexibilité locale")
print("  - Utilisation de la validation croisée pour optimiser k")

# Grid search pour trouver le meilleur k
param_grid_knn = {'n_neighbors': [3, 5, 7, 9, 11, 13, 15, 20, 25]}
knn_base = KNeighborsClassifier()

print("\nRecherche de la meilleure valeur de k...")
knn_grid = GridSearchCV(knn_base, param_grid_knn, cv=5, scoring='accuracy', 
                        n_jobs=-1, verbose=1)
knn_grid.fit(X_train, y_train)

print(f"Meilleur k: {knn_grid.best_params_['n_neighbors']}")
print(f"Score de validation croisée: {knn_grid.best_score_:.4f}")

# Afficher les scores pour différents k
print("\nRésultats pour différents k:")
for k in param_grid_knn['n_neighbors']:
    knn_temp = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn_temp, X_train, y_train, cv=5)
    print(f"  k={k:2d}: {scores.mean():.4f} (+/- {scores.std()*2:.4f})")

results['KNN'] = evaluate_model(knn_grid.best_estimator_, 
                                f"KNN (k={knn_grid.best_params_['n_neighbors']})",
                                X_train, X_test, y_train, y_test)

###############################################################################
# 5. RANDOM FOREST
###############################################################################

print("\n" + "="*80)
print("5. RANDOM FOREST")
print("="*80)

print("\nJustification de Random Forest:")
print("  - Ensemble de plusieurs arbres de décision")
print("  - Réduit le sur-apprentissage par le bagging (moyenne de plusieurs arbres)")
print("  - Captive les interactions non-linéaires entre variables")
print("  - Fournit des mesures d'importance des variables")
print("  - Peut gérer automatiquement les corrélations entre variables")
print("\nCalibration des hyperparamètres:")
print("  - n_estimators: nombre d'arbres (plus = mieux mais plus lent)")
print("  - max_depth: profondeur maximale des arbres")
print("  - min_samples_split: nombre minimum d'échantillons pour diviser un nœud")

# Grid search pour Random Forest
param_grid_rf = {
    'n_estimators': [50, 100, 200],
    'max_depth': [5, 10, 15, None],
    'min_samples_split': [2, 5, 10]
}

rf_base = RandomForestClassifier(random_state=42)

print("\nRecherche des meilleurs hyperparamètres pour Random Forest...")
rf_grid = GridSearchCV(rf_base, param_grid_rf, cv=5, scoring='accuracy',
                       n_jobs=-1, verbose=1)
rf_grid.fit(X_train, y_train)

print(f"Meilleurs paramètres: {rf_grid.best_params_}")
print(f"Score de validation croisée: {rf_grid.best_score_:.4f}")

results['Random Forest'] = evaluate_model(
    rf_grid.best_estimator_, 
    f"Random Forest (n_estimators={rf_grid.best_params_['n_estimators']})",
    X_train, X_test, y_train, y_test
)

# Importance des variables
feature_importance = pd.DataFrame({
    'Variable': X_train.columns,
    'Importance': rf_grid.best_estimator_.feature_importances_
}).sort_values('Importance', ascending=False)

print(f"\nTop 15 variables les plus importantes selon Random Forest:")
print(feature_importance.head(15).to_string(index=False))

###############################################################################
# COMPARAISON DES ALGORITHMES
###############################################################################

print("\n" + "="*80)
print("COMPARAISON DES ALGORITHMES")
print("="*80)

# Créer un DataFrame avec les résultats
comparison_df = pd.DataFrame({
    'Algorithme': [r['name'] for r in results.values()],
    'Précision Train': [r['train_score'] for r in results.values()],
    'Précision Test': [r['test_score'] for r in results.values()],
})

print("\n" + comparison_df.to_string(index=False))

# Graphique de comparaison
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Scores
algorithms = [r['name'] for r in results.values()]
train_scores = [r['train_score'] for r in results.values()]
test_scores = [r['test_score'] for r in results.values()]

x_pos = np.arange(len(algorithms))
width = 0.35

ax1.bar(x_pos - width/2, train_scores, width, label='Train', alpha=0.8)
ax1.bar(x_pos + width/2, test_scores, width, label='Test', alpha=0.8)
ax1.set_xlabel('Algorithme')
ax1.set_ylabel('Précision')
ax1.set_title('Comparaison des algorithmes')
ax1.set_xticks(x_pos)
ax1.set_xticklabels(algorithms, rotation=45, ha='right')
ax1.legend()
ax1.grid(axis='y', alpha=0.3)
ax1.set_ylim([min(min(train_scores), min(test_scores)) - 0.1, 1.0])

# Courbes ROC
ax2.plot([0, 1], [0, 1], 'k--', label='Classifieur aléatoire')
ax2.set_xlabel('False Positive Rate')
ax2.set_ylabel('True Positive Rate')
ax2.set_title('Courbes ROC')
ax2.legend()
ax2.grid(alpha=0.3)

for name, result in results.items():
    model = result['model']
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    ax2.plot(fpr, tpr, label=f'{name} (AUC={roc_auc:.3f})')

ax2.legend()
plt.tight_layout()
plt.savefig('comparison_algorithms.png', dpi=300)
plt.show()

print("\nAUC (Area Under Curve) par algorithme:")
for name, result in results.items():
    model = result['model']
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    print(f"  {name:25s}: {roc_auc:.4f}")

# Trouver le meilleur algorithme
best_model_name = max(results.keys(), key=lambda k: results[k]['test_score'])
best_model = results[best_model_name]

print(f"\n{'='*80}")
print(f"MEILLEUR ALGORITHME: {best_model_name}")
print(f"{'='*80}")
print(f"Précision sur test: {best_model['test_score']:.4f}")
print(f"Précision sur train: {best_model['train_score']:.4f}")

###############################################################################
# CONCLUSION ET ANALYSE
###############################################################################

print("\n" + "="*80)
print("CONCLUSION")
print("="*80)

print("""
ANALYSE DES VARIABLES LES PLUS PERTINENTES:
""")

# Combiner les informations de différentes méthodes
print("Variables importantes selon différentes méthodes:\n")

print("1. Par différence de moyennes (étude préliminaire):")
print("   - concave_points_worst, texture_mean, concave_points_se")
print("   - smoothness_se, radius_worst, concavity_worst")

print("\n2. Par régression logistique (coefficients):")
logit_top = coef_df.sort_values('Abs_Coefficient', ascending=False).head(5)
for idx, (_, row) in enumerate(logit_top.iterrows(), 1):
    print(f"   {idx}. {row['Variable']} (coef: {row['Coefficient']:.3f})")

print("\n3. Par Random Forest (importance):")
rf_top = feature_importance.head(5)
for idx, (_, row) in enumerate(rf_top.iterrows(), 1):
    print(f"   {idx}. {row['Variable']} (importance: {row['Importance']:.4f})")

print("""
INTERPRÉTATION:

Les variables les plus pertinentes pour distinguer les tumeurs malignes des 
tumeurs bénignes incluent principalement:

1. **concave_points_worst**: Mesure de la sévérité des parties concaves 
   (valeurs limites les plus élevées). Les cellules malignes ont souvent
   des contours plus irréguliers avec plus de concavités.

2. **texture_mean**: Texture moyenne (écart-type des valeurs de niveaux de gris).
   Les cellules malignes ont généralement une texture plus hétérogène.

3. **radius_worst, perimeter_worst, area_worst**: Les "pires" valeurs 
   (les 3 plus grandes) de rayon, périmètre et aire. Les tumeurs malignes
   sont généralement plus grandes.

4. **concavity_worst, concave_points_se**: Autres mesures de concavité.
   Les cellules malignes ont des contours plus complexes et irréguliers.

5. **smoothness_worst**: Mesure la variation locale des rayons.
   Les cellules malignes sont généralement moins lisses.

La combinaison de ces variables permet de créer des frontières de décision
qui séparent efficacement les deux classes. Les algorithmes comme Random Forest
et KNN peuvent capturer ces relations complexes, même non-linéaires.

""")

print("RECOMMANDATIONS:")
print(f"\n  → Algorithme recommandé: {best_model_name}")
print(f"     Précision: {best_model['test_score']*100:.2f}%")
print("\n  → Pour ce problème médical:")
print("     - Un taux d'erreur bas est crucial (classification incorrecte = faux positif/négatif)")
print("     - Random Forest fournit aussi l'importance des variables (interprétabilité)")
print("     - LDA et Logistique sont plus simples et rapides si interprétabilité est prioritaire")
print("\n  → Classes déséquilibrées mais modérément:")
print("     - La stratification train/test maintient la distribution")
print("     - Pas de sur-échantillonnage nécessaire (ratio ~1.7)")
print("\n  → Visualisations générées:")
print("     - class_distribution.png: Répartition des classes")
print("     - correlation_matrix.png: Corrélations entre variables")
print("     - comparison_algorithms.png: Comparaison des performances")

print("\n" + "="*80)
print("ANALYSE TERMINÉE")
print("="*80)

