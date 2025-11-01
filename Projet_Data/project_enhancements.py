# Breast Cancer Wisconsin Dataset - Enhanced Analysis
# Implementing advanced techniques: Ensemble, Feature Selection, PCA, Cost-Sensitive, Neural Networks, Cross-Validation

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, KFold, StratifiedKFold
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, BaggingClassifier, AdaBoostClassifier
from sklearn.feature_selection import SelectKBest, f_classif, chi2, mutual_info_classif
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, roc_curve, auc, roc_auc_score, recall_score
from sklearn.neural_network import MLPClassifier
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("BREAST CANCER WISCONSIN DATASET - ENHANCED ANALYSIS")
print("="*80)

# 1. Load and prepare data
print("\n1. LOADING DATA")
print("-"*80)
df = pd.read_csv('wdbc.data', header=None)

# Create column names
feature_names = ['ID', 'Diagnosis']
feat = ['radius', 'texture', 'perimeter', 'area', 'smoothness', 'compactness',
        'concavity', 'concave_points', 'symmetry', 'fractal_dimension']
features = []
for f in feat:
    features.append(f + '_mean')
    features.append(f + '_se')
    features.append(f + '_worst')

feature_names.extend(features)
df.columns = feature_names

X = df.drop(['ID', 'Diagnosis'], axis=1)
y = df['Diagnosis'].map({'B': 0, 'M': 1})

print(f"Dataset shape: {X.shape}")
print(f"Class distribution: {y.value_counts().to_dict()}")

# Standardize features
scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

# Split train/test
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.3, random_state=42, stratify=y
)

print(f"\nTrain: {X_train.shape}, Test: {X_test.shape}")
print(f"Train class distribution: {y_train.value_counts().to_dict()}")
print(f"Test class distribution: {y_test.value_counts().to_dict()}")

# Define class weights for Random Forest to optimize recall (minimize false negatives)
class_weight_rf = {0: 1, 1: 3}  # Penalize false negatives 3x more than false positives

###############################################################################
# ENHANCEMENT 1: CROSS-VALIDATION EVALUATION
###############################################################################

print("\n" + "="*80)
print("ENHANCEMENT 1: K-FOLD CROSS-VALIDATION EVALUATION")
print("="*80)

# Perform 10-fold cross-validation on various algorithms
algorithms = {
    'LDA': LinearDiscriminantAnalysis(solver='svd'),
    'QDA': QuadraticDiscriminantAnalysis(),
    'Logistic': LogisticRegression(max_iter=2000, random_state=42),
    'KNN': KNeighborsClassifier(n_neighbors=7),
    'Random Forest': RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42, class_weight=class_weight_rf)
}

cv_results = {}
kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

print("\nPerforming 10-fold cross-validation...")
for name, model in algorithms.items():
    scores = cross_val_score(model, X_scaled, y, cv=kfold, scoring='accuracy')
    cv_results[name] = {
        'mean': scores.mean(),
        'std': scores.std(),
        'scores': scores
    }
    print(f"{name:20s}: {scores.mean():.4f} (+/- {scores.std()*2:.4f})")

# Visualize CV results
fig, ax = plt.subplots(figsize=(10, 6))
models = list(cv_results.keys())
means = [cv_results[m]['mean'] for m in models]
stds = [cv_results[m]['std'] for m in models]

ax.barh(models, means, xerr=stds, capsize=5)
ax.set_xlabel('Accuracy')
ax.set_title('10-Fold Cross-Validation Performance')
ax.set_xlim([0.90, 1.0])
ax.grid(axis='x', alpha=0.3)

for i, (mean, std) in enumerate(zip(means, stds)):
    ax.text(mean + std + 0.001, i, f'{mean:.4f}', va='center')

plt.tight_layout()
plt.savefig('cv_results.png', dpi=300)
plt.close()

print("\n✓ Cross-validation results saved to cv_results.png")
print("Best CV performance:", max(cv_results.items(), key=lambda x: x[1]['mean']))

###############################################################################
# ENHANCEMENT 2: FEATURE SELECTION
###############################################################################

print("\n" + "="*80)
print("ENHANCEMENT 2: FEATURE SELECTION")
print("="*80)

# Method 1: SelectKBest with f_classif (ANOVA F-value)
selector_f = SelectKBest(score_func=f_classif, k=10)
X_train_selected_f = selector_f.fit_transform(X_train, y_train)
X_test_selected_f = selector_f.transform(X_test)
selected_features_f = X_train.columns[selector_f.get_support()]

print("\nTop 10 features selected by ANOVA F-test:")
for i, feat in enumerate(selected_features_f, 1):
    print(f"  {i:2d}. {feat}")

# Method 2: Mutual Information
selector_mi = SelectKBest(score_func=mutual_info_classif, k=10)
X_train_selected_mi = selector_mi.fit_transform(X_train, y_train)
X_test_selected_mi = selector_mi.transform(X_test)
selected_features_mi = X_train.columns[selector_mi.get_support()]

print("\nTop 10 features selected by Mutual Information:")
for i, feat in enumerate(selected_features_mi, 1):
    print(f"  {i:2d}. {feat}")

# Evaluate models with feature selection
print("\nEvaluating models with feature selection...")

def evaluate_with_selection(X_tr, X_te, y_tr, y_te, feature_set_name):
    """Evaluate multiple models with selected features"""
    results = {}
    
    models = {
        'LDA': LinearDiscriminantAnalysis(solver='svd'),
        'Logistic': LogisticRegression(max_iter=2000, random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42, class_weight=class_weight_rf)
    }
    
    for name, model in models.items():
        model.fit(X_tr, y_tr)
        test_score = model.score(X_te, y_te)
        results[name] = test_score
    
    return results

# Evaluate with all features (baseline)
results_all = evaluate_with_selection(X_train, X_test, y_train, y_test, "All Features")
results_f = evaluate_with_selection(X_train_selected_f, X_test_selected_f, y_train, y_test, "ANOVA F-test")
results_mi = evaluate_with_selection(X_train_selected_mi, X_test_selected_mi, y_train, y_test, "Mutual Information")

# Compare results
print("\n" + "-"*80)
print("FEATURE SELECTION COMPARISON")
print("-"*80)

comparison_df = pd.DataFrame({
    'All Features': results_all,
    'ANOVA F-test (10)': results_f,
    'Mutual Info (10)': results_mi
})

print(comparison_df.to_string())

# Visualize comparison
fig, ax = plt.subplots(figsize=(12, 6))
x = np.arange(len(comparison_df.index))
width = 0.25

ax.bar(x - width, comparison_df['All Features'], width, label='All Features', alpha=0.8)
ax.bar(x, comparison_df['ANOVA F-test (10)'], width, label='ANOVA F-test (10)', alpha=0.8)
ax.bar(x + width, comparison_df['Mutual Info (10)'], width, label='Mutual Info (10)', alpha=0.8)

ax.set_ylabel('Test Accuracy')
ax.set_title('Feature Selection Comparison')
ax.set_xticks(x)
ax.set_xticklabels(comparison_df.index, rotation=45, ha='right')
ax.legend()
ax.grid(axis='y', alpha=0.3)
ax.set_ylim([0.9, 1.0])

plt.tight_layout()
plt.savefig('feature_selection_comparison.png', dpi=300)
plt.close()

print("\n✓ Feature selection comparison saved to feature_selection_comparison.png")

###############################################################################
# ENHANCEMENT 3: DIMENSIONALITY REDUCTION (PCA)
###############################################################################

print("\n" + "="*80)
print("ENHANCEMENT 3: DIMENSIONALITY REDUCTION (PCA)")
print("="*80)

# Apply PCA
pca = PCA()
X_train_pca_full = pca.fit_transform(X_train)
X_test_pca_full = pca.transform(X_test)

# Analyze explained variance
cumulative_variance = np.cumsum(pca.explained_variance_ratio_)

print("\nExplained variance by component:")
for i in range(min(10, len(pca.explained_variance_ratio_))):
    print(f"  PC{i+1}: {pca.explained_variance_ratio_[i]:.4f} (cumulative: {cumulative_variance[i]:.4f})")

# Find number of components for 95% variance
n_components_95 = np.argmax(cumulative_variance >= 0.95) + 1
print(f"\nComponents for 95% variance: {n_components_95}")

# Visualize explained variance
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

ax1.plot(range(1, len(cumulative_variance) + 1), 
         pca.explained_variance_ratio_, 'o-', linewidth=2, markersize=6)
ax1.set_xlabel('Principal Component')
ax1.set_ylabel('Explained Variance Ratio')
ax1.set_title('Explained Variance by Component')
ax1.grid(alpha=0.3)
ax1.axvline(n_components_95, color='r', linestyle='--', label=f'95% variance ({n_components_95} components)')
ax1.legend()

ax2.plot(range(1, len(cumulative_variance) + 1), 
         cumulative_variance, 'o-', linewidth=2, markersize=6)
ax2.set_xlabel('Principal Component')
ax2.set_ylabel('Cumulative Explained Variance')
ax2.set_title('Cumulative Explained Variance')
ax2.axhline(0.95, color='r', linestyle='--', label='95% threshold')
ax2.axhline(0.90, color='orange', linestyle='--', label='90% threshold')
ax2.grid(alpha=0.3)
ax2.legend()

plt.tight_layout()
plt.savefig('pca_analysis.png', dpi=300)
plt.close()

# Evaluate models with different numbers of PCA components
print("\nEvaluating models with different PCA components...")

pca_n_components = [5, 10, 15, 20, n_components_95]
pca_results = {n: {} for n in pca_n_components}

for n in pca_n_components:
    pca_temp = PCA(n_components=n)
    X_train_pca = pca_temp.fit_transform(X_train)
    X_test_pca = pca_temp.transform(X_test)
    
    models_to_test = {
        'LDA': LinearDiscriminantAnalysis(solver='svd'),
        'Logistic': LogisticRegression(max_iter=2000, random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42, class_weight=class_weight_rf)
    }
    
    for name, model in models_to_test.items():
        model.fit(X_train_pca, y_train)
        score = model.score(X_test_pca, y_test)
        pca_results[n][name] = score

pca_comparison_df = pd.DataFrame(pca_results).T
print("\n" + "-"*80)
print("PCA COMPONENT ANALYSIS")
print("-"*80)
print(pca_comparison_df.to_string())

# Visualize PCA results
fig, ax = plt.subplots(figsize=(12, 6))
for model in pca_comparison_df.columns:
    ax.plot(pca_comparison_df.index, pca_comparison_df[model], 'o-', label=model, linewidth=2, markersize=8)

ax.set_xlabel('Number of PCA Components')
ax.set_ylabel('Test Accuracy')
ax.set_title('Model Performance with Different PCA Components')
ax.legend()
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('pca_model_comparison.png', dpi=300)
plt.close()

print("\n✓ PCA analysis saved to pca_analysis.png and pca_model_comparison.png")

###############################################################################
# ENHANCEMENT 4: ENSEMBLE METHODS
###############################################################################

print("\n" + "="*80)
print("ENHANCEMENT 4: ENSEMBLE METHODS")
print("="*80)

# Train base models
lda = LinearDiscriminantAnalysis(solver='svd')
logistic = LogisticRegression(max_iter=2000, random_state=42)
# Random Forest optimized for recall (minimize false negatives) - higher weight for malignant class
rf = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42, class_weight=class_weight_rf)
knn = KNeighborsClassifier(n_neighbors=7)

print("\nTraining base models...")
base_models = {'LDA': lda, 'Logistic': logistic, 'RF': rf, 'KNN': knn}
base_scores = {}

for name, model in base_models.items():
    model.fit(X_train, y_train)
    score = model.score(X_test, y_test)
    base_scores[name] = score
    print(f"  {name}: {score:.4f}")

# Plot Random Forest feature importances (balanced)
try:
    importances = rf.feature_importances_
    sorted_indices = np.argsort(importances)[::-1]
    plt.figure(figsize=(10, 8))
    plt.barh(range(len(importances)), importances[sorted_indices], align='center')
    plt.yticks(range(len(importances)), X_train.columns[sorted_indices])
    plt.gca().invert_yaxis()
    plt.xlabel('Importance')
    plt.title('Random Forest Feature Importances (Recall-Optimized)')
    plt.tight_layout()
    plt.savefig('Features_Importance_RandomForest.png', dpi=300)
    plt.close()
    print("\n✓ Random Forest feature importances saved to Features_Importance_RandomForest.png")
    
    # Report recall for Random Forest
    y_pred_rf = rf.predict(X_test)
    recall_rf = recall_score(y_test, y_pred_rf)
    cm_rf = confusion_matrix(y_test, y_pred_rf)
    print(f"  Random Forest Recall: {recall_rf:.4f}")
    print(f"  Random Forest False Negatives: {cm_rf[1,0]}")
except Exception as e:
    print(f"\n! Failed to generate Random Forest feature importances: {str(e)}")

# Method 1: Voting Classifier (Hard voting)
print("\nTraining Voting Classifier (Hard)...")
voting_hard = VotingClassifier(
    estimators=[
        ('lda', lda),
        ('logistic', logistic),
        ('rf', rf)
    ],
    voting='hard'
)
voting_hard.fit(X_train, y_train)
score_hard = voting_hard.score(X_test, y_test)
print(f"  Voting (Hard): {score_hard:.4f}")

# Method 2: Voting Classifier (Soft voting)
print("\nTraining Voting Classifier (Soft)...")
voting_soft = VotingClassifier(
    estimators=[
        ('lda', lda),
        ('logistic', logistic),
        ('rf', rf)
    ],
    voting='soft'
)
voting_soft.fit(X_train, y_train)
score_soft = voting_soft.score(X_test, y_test)
print(f"  Voting (Soft): {score_soft:.4f}")

# Method 3: Bagging Classifier
print("\nTraining Bagging Classifier...")
bagging = BaggingClassifier(
    base_estimator=LogisticRegression(max_iter=2000, random_state=42),
    n_estimators=20,
    random_state=42
)
bagging.fit(X_train, y_train)
score_bagging = bagging.score(X_test, y_test)
print(f"  Bagging: {score_bagging:.4f}")

# Method 4: AdaBoost
print("\nTraining AdaBoost...")
adaboost = AdaBoostClassifier(
    base_estimator=LogisticRegression(max_iter=2000, random_state=42),
    n_estimators=50,
    random_state=42
)
adaboost.fit(X_train, y_train)
score_adaboost = adaboost.score(X_test, y_test)
print(f"  AdaBoost: {score_adaboost:.4f}")

# Collect ensemble results
ensemble_scores = {
    'Base: LDA': base_scores['LDA'],
    'Base: Logistic': base_scores['Logistic'],
    'Base: Random Forest': base_scores['RF'],
    'Base: KNN': base_scores['KNN'],
    'Voting (Hard)': score_hard,
    'Voting (Soft)': score_soft,
    'Bagging': score_bagging,
    'AdaBoost': score_adaboost
}

# Visualize ensemble results
fig, ax = plt.subplots(figsize=(12, 6))
models_list = list(ensemble_scores.keys())
scores_list = list(ensemble_scores.values())

colors = ['skyblue'] * 4 + ['lightcoral'] * 4
ax.barh(models_list, scores_list, color=colors, alpha=0.8)

ax.set_xlabel('Test Accuracy')
ax.set_title('Ensemble Methods Comparison')
ax.set_xlim([0.90, 1.0])
ax.grid(axis='x', alpha=0.3)

for i, score in enumerate(scores_list):
    ax.text(score + 0.001, i, f'{score:.4f}', va='center')

plt.tight_layout()
plt.savefig('ensemble_comparison.png', dpi=300)
plt.close()

print("\n✓ Ensemble comparison saved to ensemble_comparison.png")
print("\nBest ensemble method:", max(ensemble_scores.items(), key=lambda x: x[1]))

###############################################################################
# ENHANCEMENT 5: COST-SENSITIVE LEARNING
###############################################################################

print("\n" + "="*80)
print("ENHANCEMENT 5: COST-SENSITIVE LEARNING")
print("="*80)
print("\nNote: In medical diagnosis, False Negatives (missing malignant cases)")
print("      are more critical than False Positives (incorrectly predicting benign).")

# Define cost matrix
# Cost of misclassifying malignant as benign (FN) is higher than benign as malignant (FP)
cost_false_negative = 5  # Cost of missing a malignant tumor
cost_false_positive = 1  # Cost of incorrectly predicting benign as malignant

print(f"\nCost matrix:")
print(f"  False Negative (FN): {cost_false_negative}")
print(f"  False Positive (FP): {cost_false_positive}")

# Custom scoring function
def custom_cost_score(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    # cm[i,j] = actual class i, predicted class j
    # For binary: cm[0,0]=TN, cm[0,1]=FP, cm[1,0]=FN, cm[1,1]=TP
    total_cost = cost_false_negative * cm[1, 0] + cost_false_positive * cm[0, 1]
    return -total_cost  # Negative because higher is better

# Method 1: Adjust class weights in Logistic Regression
print("\nTraining Logistic Regression with class weights...")
logistic_weighted = LogisticRegression(
    class_weight={0: 1, 1: cost_false_negative},
    max_iter=2000,
    random_state=42
)
logistic_weighted.fit(X_train, y_train)
y_pred_weighted = logistic_weighted.predict(X_test)

cm_weighted = confusion_matrix(y_test, y_pred_weighted)
print(f"  Accuracy: {accuracy_score(y_test, y_pred_weighted):.4f}")
print(f"  Confusion Matrix:")
print(f"     [{cm_weighted[0,0]:3d}  {cm_weighted[0,1]:3d}]")
print(f"     [{cm_weighted[1,0]:3d}  {cm_weighted[1,1]:3d}]")
print(f"  Cost: {cost_false_negative * cm_weighted[1,0] + cost_false_positive * cm_weighted[0,1]}")

# Compare with standard logistic regression
logistic_standard = LogisticRegression(max_iter=2000, random_state=42)
logistic_standard.fit(X_train, y_train)
y_pred_standard = logistic_standard.predict(X_test)

cm_standard = confusion_matrix(y_test, y_pred_standard)
print(f"\nStandard Logistic Regression:")
print(f"  Accuracy: {accuracy_score(y_test, y_pred_standard):.4f}")
print(f"  Confusion Matrix:")
print(f"     [{cm_standard[0,0]:3d}  {cm_standard[0,1]:3d}]")
print(f"     [{cm_standard[1,0]:3d}  {cm_standard[1,1]:3d}]")
print(f"  Cost: {cost_false_negative * cm_standard[1,0] + cost_false_positive * cm_standard[0,1]}")

# Visualize cost-sensitive comparison
fig, ax = plt.subplots(figsize=(10, 6))
categories = ['Standard\nAccuracy', 'Weighted\nAccuracy', 'Standard\nCost', 'Weighted\nCost']
values = [
    accuracy_score(y_test, y_pred_standard),
    accuracy_score(y_test, y_pred_weighted),
    500 - (cost_false_negative * cm_standard[1,0] + cost_false_positive * cm_standard[0,1]),  # Invert for display
    500 - (cost_false_negative * cm_weighted[1,0] + cost_false_positive * cm_weighted[0,1])
]

bars = ax.bar(categories, values, color=['skyblue', 'lightcoral', 'lightgreen', 'orange'], alpha=0.8)
ax.set_ylabel('Score (Normalized Cost)')
ax.set_title('Cost-Sensitive Learning Comparison')
ax.grid(axis='y', alpha=0.3)

for i, (bar, val) in enumerate(zip(bars, values)):
    if i < 2:
        ax.text(bar.get_x() + bar.get_width()/2, val + 1, f'{val:.4f}', 
                ha='center', va='bottom')
    else:
        ax.text(bar.get_x() + bar.get_width()/2, val + 1, f'{val:.0f}', 
                ha='center', va='bottom')

plt.tight_layout()
plt.savefig('cost_sensitive_comparison.png', dpi=300)
plt.close()

print("\n✓ Cost-sensitive comparison saved to cost_sensitive_comparison.png")

###############################################################################
# ENHANCEMENT 6: DEEP LEARNING (NEURAL NETWORKS)
###############################################################################

print("\n" + "="*80)
print("ENHANCEMENT 6: DEEP LEARNING (NEURAL NETWORKS)")
print("="*80)

# Define neural network architectures
print("\nTraining neural networks with different architectures...")

nn_configs = {
    'Small (20)': MLPClassifier(hidden_layer_sizes=(20,), max_iter=2000, random_state=42),
    'Medium (50)': MLPClassifier(hidden_layer_sizes=(50,), max_iter=2000, random_state=42),
    'Large (100)': MLPClassifier(hidden_layer_sizes=(100,), max_iter=2000, random_state=42),
    'Deep (50, 20)': MLPClassifier(hidden_layer_sizes=(50, 20), max_iter=2000, random_state=42),
    'Deeper (100, 50)': MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=2000, random_state=42)
}

nn_results = {}

for name, model in nn_configs.items():
    try:
        model.fit(X_train, y_train)
        score = model.score(X_test, y_test)
        nn_results[name] = score
        print(f"  {name:20s}: {score:.4f}")
    except Exception as e:
        print(f"  {name:20s}: Failed - {str(e)}")

if len(nn_results) > 0:
    # Visualize neural network results
    fig, ax = plt.subplots(figsize=(12, 6))
    networks = list(nn_results.keys())
    scores = list(nn_results.values())
    
    ax.barh(networks, scores, alpha=0.8)
    ax.set_xlabel('Test Accuracy')
    ax.set_title('Neural Network Architectures Comparison')
    ax.set_xlim([0.90, 1.0])
    ax.grid(axis='x', alpha=0.3)
    
    for i, score in enumerate(scores):
        ax.text(score + 0.001, i, f'{score:.4f}', va='center')
    
    plt.tight_layout()
    plt.savefig('neural_networks_comparison.png', dpi=300)
    plt.close()
    
    print("\n✓ Neural networks comparison saved to neural_networks_comparison.png")
    
    best_nn = max(nn_results.items(), key=lambda x: x[1])
    print(f"\nBest neural network: {best_nn[0]} with accuracy {best_nn[1]:.4f}")

print("\n" + "="*80)
print("ENHANCED ANALYSIS COMPLETE")
print("="*80)

###############################################################################
# ENHANCEMENT 7: XGBOOST EVALUATION (TEST SIZE = 0.2)
###############################################################################

print("\n" + "="*80)
print("ENHANCEMENT 7: XGBOOST (test_size=0.2) - Recall-Optimized with AUC & Confusion Matrix")
print("="*80)

try:
    from xgboost import XGBClassifier

    # Create a dedicated split for XGBoost with test_size=0.2
    X_train_xgb, X_test_xgb, y_train_xgb, y_test_xgb = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )

    # Calculate scale_pos_weight to handle class imbalance and optimize for recall
    # scale_pos_weight = negative_samples / positive_samples
    neg_count = (y_train_xgb == 0).sum()
    pos_count = (y_train_xgb == 1).sum()
    scale_pos_weight = neg_count / pos_count
    
    print(f"\nClass distribution - Negative: {neg_count}, Positive: {pos_count}")
    print(f"Using scale_pos_weight: {scale_pos_weight:.2f} to optimize for recall")

    xgb_model = XGBClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=3,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        scale_pos_weight=scale_pos_weight * 1.5,  # Multiply by 1.5 to further penalize false negatives
        random_state=42,
        n_jobs=-1,
        eval_metric='logloss',
        use_label_encoder=False
    )

    print("\nTraining XGBoost (recall-optimized)...")
    xgb_model.fit(X_train_xgb, y_train_xgb)

    # Get probability predictions
    y_proba_xgb = xgb_model.predict_proba(X_test_xgb)[:, 1]
    
    # Find optimal threshold that maximizes recall
    thresholds = np.arange(0.1, 0.6, 0.05)
    best_recall = 0
    best_threshold = 0.5
    
    for threshold in thresholds:
        y_pred_thresh = (y_proba_xgb >= threshold).astype(int)
        recall = recall_score(y_test_xgb, y_pred_thresh)
        if recall > best_recall:
            best_recall = recall
            best_threshold = threshold
    
    # Use optimal threshold for final predictions
    y_pred_xgb = (y_proba_xgb >= best_threshold).astype(int)

    auc_xgb = roc_auc_score(y_test_xgb, y_proba_xgb)
    recall_xgb = recall_score(y_test_xgb, y_pred_xgb)
    cm_xgb = confusion_matrix(y_test_xgb, y_pred_xgb)

    print(f"\n  Optimal threshold: {best_threshold:.2f}")
    print(f"  AUC: {auc_xgb:.4f}")
    print(f"  Recall: {recall_xgb:.4f}")
    print("  Confusion Matrix:")
    print(f"     [{cm_xgb[0,0]:3d}  {cm_xgb[0,1]:3d}]")
    print(f"     [{cm_xgb[1,0]:3d}  {cm_xgb[1,1]:3d}]")
    print(f"  False Negatives: {cm_xgb[1,0]}")

except ImportError:
    print("XGBoost is not installed. Please install it with 'pip install xgboost'.")
except Exception as e:
    print(f"XGBoost evaluation failed: {str(e)}")

###############################################################################
# SUMMARY OF ALL RESULTS
###############################################################################

print("\n" + "="*80)
print("SUMMARY OF ALL ENHANCEMENTS")
print("="*80)

print("\n1. CROSS-VALIDATION RESULTS:")
print("-"*80)
for name, result in sorted(cv_results.items(), key=lambda x: x[1]['mean'], reverse=True):
    print(f"{name:20s}: {result['mean']:.4f} +/- {result['std']:.4f}")

print("\n2. FEATURE SELECTION:")
print("-"*80)
print("Best with ANOVA F-test (10 features):")
for name, score in sorted(results_f.items(), key=lambda x: x[1], reverse=True):
    print(f"  {name}: {score:.4f}")

print("\n3. PCA ANALYSIS:")
print("-"*80)
print(f"Optimal components for 95% variance: {n_components_95}")
if len(pca_comparison_df) > 0:
    best_pca_config = pca_comparison_df.stack().idxmax()
    best_pca_score = pca_comparison_df.max().max()
    print(f"Best PCA configuration: {best_pca_config[0]} components with {best_pca_config[1]} ({best_pca_score:.4f})")

print("\n4. ENSEMBLE METHODS:")
print("-"*80)
for name, score in sorted(ensemble_scores.items(), key=lambda x: x[1], reverse=True):
    print(f"{name:25s}: {score:.4f}")

print("\n5. COST-SENSITIVE LEARNING:")
print("-"*80)
print(f"Weighted Logistic Regression:")
print(f"  Accuracy: {accuracy_score(y_test, y_pred_weighted):.4f}")
print(f"  False Negatives: {cm_weighted[1,0]}")
print(f"  Total Cost: {cost_false_negative * cm_weighted[1,0] + cost_false_positive * cm_weighted[0,1]}")

if len(nn_results) > 0:
    print("\n6. NEURAL NETWORKS:")
    print("-"*80)
    for name, score in sorted(nn_results.items(), key=lambda x: x[1], reverse=True):
        print(f"  {name:20s}: {score:.4f}")

print("\n" + "="*80)
print("ANALYSIS COMPLETE - All visualizations saved")
print("="*80)

