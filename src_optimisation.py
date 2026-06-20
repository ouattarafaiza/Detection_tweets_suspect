# src_optimisation.py
import joblib
import pandas as pd
from sklearn.model_selection import GridSearchCV, train_test_split

print("=== Partie 6 : Optimisation des Hyperparamètres (Grid Search) ===")

# 1. Chargement des données brutes vectorisées
X_tfidf, y = joblib.load('matrix_tfidf.pkl')
modeles = joblib.load('modeles_construits.pkl')

# Récupération
rf_modele = modeles['Random_Forest']

# Utilisation du même split d'entraînement pour la cohérence
X_train, _, y_train, _ = train_test_split(
    X_tfidf, y, test_size=0.2, random_state=42, stratify=y
)

# 2. Définition de la grille des paramètres à tester
grille_parametres = {
    'n_estimators': [50, 100],
    'max_depth': [10, 20, None]
}

print("Lancement du Grid Search")
# cv=3 et scoring='f1' pour privilégier l'équilibre des classes
grid_search = GridSearchCV(estimator=rf_modele, param_grid=grille_parametres, cv=3, scoring='f1', n_jobs=-1)
grid_search.fit(X_train, y_train)

# 3. Affichage et sauvegarde des résultats
print("\n=== RÉSULTATS DE L'OPTIMISATION ===")
print(f"Meilleurs paramètres trouvés : {grid_search.best_params_}")
print(f"Meilleur score F1 associé : {grid_search.best_score_:.4f}")

# Sauvegarde des paramètres
with open("parametres_optimaux.txt", "w") as f:
    f.write(f"Meilleurs parametres : {grid_search.best_params_}\n")
    f.write(f"Meilleur score F1 : {grid_search.best_score_:.4f}\n")

# Sauvegarde du modèle final optimisé
joblib.dump(grid_search.best_estimator_, 'modele_final_optimise.pkl')
print("\nFichiers 'parametres_optimaux.txt' et 'modele_final_optimise.pkl' sauvegardés.")