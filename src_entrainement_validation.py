import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate

print("=== Partie 5 : Entraînement et Validation Croisée ===")

# 1. Chargement des artefacts des étapes précédentes
X_tfidf, y = joblib.load('matrix_tfidf.pkl')
modeles = joblib.load('modeles_construits.pkl')

# 2. Séparation des données en ensembles d'entraînement (80%) et de test (20%)
# L'argument 'stratify=y' maintient la proportion 90/10 dans le train et le test
X_train, X_test, y_train, y_test = train_test_split(
    X_tfidf, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Taille de l'ensemble d'entraînement : {X_train.shape[0]} tweets")
print(f"Taille de l'ensemble de test : {X_test.shape[0]} tweets\n")

# 3. Configuration de la validation croisée stratifiée (3 plis)
cv_stratifie = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
metriques_minimales = ['accuracy', 'precision', 'recall', 'f1']
performances_cv = {}

# 4. Entraînement et évaluation par validation croisée pour chaque modèle
for nom, modele in modeles.items():
    print(f"Calcul de la validation croisée pour : {nom}...")
    scores = cross_validate(modele, X_train, y_train, cv=cv_stratifie, scoring=metriques_minimales, n_jobs=-1)
    
    # Stockage des moyennes des scores obtenus sur les plis de validation
    performances_cv[nom] = {
        'Accuracy': np.mean(scores['test_accuracy']),
        'Precision': np.mean(scores['test_precision']),
        'Recall': np.mean(scores['test_recall']),
        'F1-Score': np.mean(scores['test_f1'])
    }

# 5. Présentation du tableau comparatif des métriques obtenues
df_performances = pd.DataFrame(performances_cv).T
print("\n=== TABLEAU COMPARATIF DES PERFORMANCES (VALIDATION CROISÉE) ===")
print(df_performances.to_string())

# Sauvegarde des résultats et des données de test pour la Partie 6
df_performances.to_csv('performances_validation.csv')
# On entraîne les modèles sur la totalité du Train pour qu'ils soient prêts pour la Partie 6
for nom, modele in modeles.items():
    modele.fit(X_train, y_train)

joblib.dump(modeles, 'modeles_entraines.pkl')
joblib.dump((X_test, y_test), 'donnees_test.pkl')
print("\nFichiers 'performances_validation.csv', 'modeles_entraines.pkl' et 'donnees_test.pkl' sauvegardés.")