import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# Modules Scikit-learn
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score

# Modules Imbalanced-learn (pour SMOTE)
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

# =========================================================
# 1. Chargement et exploration du dataset 
# =========================================================
print("--- ÉTAPE 1 : CHARGEMENT ET EXPLORATION ---")

# 1.1 Charger le fichier (gestion des espaces vides comme NaN)
df = pd.read_csv('CHD.csv', na_values=[' ', ''])

# 1.2 & 1.3 Aperçu et Infos
print("Premières lignes :\n", df.head())
print("\nTypes des variables :")
print(df.dtypes)

# 1.4 Distribution de la variable famhist (Sauvegarde image)
plt.figure(figsize=(6, 4))
sns.countplot(x='famhist', data=df)
plt.title("Distribution de la variable famhist")
plt.savefig('graph_distribution_famhist.png') # Sauvegarde
plt.close() # Ferme la figure pour libérer la mémoire
print("-> Graphique sauvegardé : 'graph_distribution_famhist.png'")

# 1.5 Heatmap des valeurs manquantes (Sauvegarde image)
plt.figure(figsize=(10, 6))
sns.heatmap(df.isnull(), cbar=False, yticklabels=False, cmap='viridis')
plt.title("Carte des valeurs manquantes")
plt.savefig('graph_missing_values.png') # Sauvegarde
plt.close()
print("-> Graphique sauvegardé : 'graph_missing_values.png'")

# =========================================================
# 2. Séparation du dataset 
# =========================================================
print("\n--- ÉTAPE 2 : SÉPARATION DU DATASET ---")

X = df.drop('chd', axis=1) # Variables explicatives
y = df['chd']              # Variable cible

# Division 33% test, random_state=123, stratification
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=123, stratify=y
)
print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")

# =========================================================
# 3, 4 & 5. Prétraitement (Pipelines) 
# =========================================================
print("\n--- ÉTAPES 3, 4 & 5 : PRÉTRAITEMENT ---")

# Pipeline Numérique : Imputation médiane + Standardisation
num_features = ['sbp', 'tobacco', 'ldl', 'adiposity', 'typea', 'obesity', 'alcohol', 'age']
num_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

# Pipeline Catégorielle : Imputation fréquente + OneHot
cat_features = ['famhist']
cat_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(drop='first')) # drop='first' évite la multicolinéarité
])

# Préprocesseur global (ColumnTransformer)
preprocessor = ColumnTransformer(
    transformers=[
        ('num', num_transformer, num_features),
        ('cat', cat_transformer, cat_features)
    ]
)

# =========================================================
# 6. Modèle Supervisé avec ACP (LogReg) 
# =========================================================
print("\n--- ÉTAPE 6 : RÉGRESSION LOGISTIQUE AVEC ACP ---")

pipe_lr_pca = Pipeline([
    ('preprocessor', preprocessor),
    ('pca', PCA()), # ACP sans restriction pour l'analyse
    ('classifier', LogisticRegression())
])

pipe_lr_pca.fit(X_train, y_train)
y_pred_pca = pipe_lr_pca.predict(X_test)

print("Rapport de classification (ACP par défaut) :")
print(classification_report(y_test, y_pred_pca))

# =========================================================
# 7. Variance expliquée par l'ACP 
# =========================================================
print("\n--- ÉTAPE 7 : ANALYSE DE LA VARIANCE ACP ---")

pca_step = pipe_lr_pca.named_steps['pca']
cum_variance = np.cumsum(pca_step.explained_variance_ratio_)

# Tracer et sauvegarder le graphique de variance
plt.figure(figsize=(8, 5))
plt.plot(range(1, len(cum_variance) + 1), cum_variance, marker='o', linestyle='--')
plt.axhline(y=0.90, color='r', linestyle='-', label='Seuil 90%')
plt.xlabel('Nombre de composantes')
plt.ylabel('Variance cumulée')
plt.title('Variance expliquée par ACP')
plt.legend()
plt.grid()
plt.savefig('graph_acp_variance.png')
plt.close()
print("-> Graphique sauvegardé : 'graph_acp_variance.png'")

# Nombre de composantes pour atteindre 90%
n_components_90 = np.argmax(cum_variance >= 0.90) + 1
print(f"Composantes nécessaires pour >= 90% de variance : {n_components_90}")

# Réentraînement avec le bon nombre de composantes
pipe_lr_pca_opt = Pipeline([
    ('preprocessor', preprocessor),
    ('pca', PCA(n_components=n_components_90)),
    ('classifier', LogisticRegression())
])
pipe_lr_pca_opt.fit(X_train, y_train)
score_pca = pipe_lr_pca_opt.score(X_test, y_test)
print(f"Accuracy (LogReg + ACP {n_components_90} comp): {score_pca:.4f}")

# =========================================================
# 8. Comparaison sans ACP 
# =========================================================
print("\n--- ÉTAPE 8 : COMPARAISON SANS ACP ---")

pipe_lr_no_pca = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression())
])
pipe_lr_no_pca.fit(X_train, y_train)
score_no_pca = pipe_lr_no_pca.score(X_test, y_test)

print(f"Accuracy SANS ACP : {score_no_pca:.4f}")
if score_no_pca > score_pca:
    print("-> Le modèle SANS ACP performe mieux.")
else:
    print("-> Le modèle AVEC ACP est meilleur ou équivalent.")

# =========================================================
# 9. Modèle KNN (SMOTE + ACP + GridSearch) 
# =========================================================
print("\n--- ÉTAPE 9 : KNN OPTIMISÉ (AVEC SMOTE) ---")

# Utilisation de ImbPipeline pour inclure SMOTE correctement
pipe_knn = ImbPipeline([
    ('preprocessor', preprocessor),
    ('smote', SMOTE(random_state=42)),
    ('pca', PCA(n_components=n_components_90)),
    ('knn', KNeighborsClassifier())
])

# Optimisation de k (n_neighbors)
param_grid = {'knn__n_neighbors': range(1, 21)}
grid_knn = GridSearchCV(pipe_knn, param_grid, cv=5, scoring='accuracy')
grid_knn.fit(X_train, y_train)

print(f"Meilleurs paramètres KNN : {grid_knn.best_params_}")
best_knn = grid_knn.best_estimator_
acc_knn = best_knn.score(X_test, y_test)
print(f"Accuracy KNN Optimisé : {acc_knn:.4f}")

# =========================================================
# 10. Entraînement Final et Sauvegarde 
# =========================================================
print("\n--- ÉTAPE 10 : SAUVEGARDE DU MEILLEUR MODÈLE ---")

# Comparaison finale des 3 approches
results = {
    'LogReg_ACP': score_pca,
    'LogReg_NoACP': score_no_pca,
    'KNN_Smote': acc_knn
}
best_model_name = max(results, key=results.get)
print(f"Meilleur modèle global : {best_model_name} (Acc: {results[best_model_name]:.4f})")

# Sélection de l'objet modèle correspondant
if best_model_name == 'KNN_Smote':
    final_model = best_knn
elif best_model_name == 'LogReg_NoACP':
    final_model = pipe_lr_no_pca
else:
    final_model = pipe_lr_pca_opt

# Entraînement sur TOUTES les données (X, y) avant déploiement
final_model.fit(X, y)

# Sauvegarde
joblib.dump(final_model, 'Model.pkl')
print("Modèle final sauvegardé sous 'Model.pkl'.")