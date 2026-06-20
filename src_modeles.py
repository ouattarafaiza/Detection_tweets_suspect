import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier

print("=== Partie 4 : Construction des modèles ===")

# 1. Gestion du déséquilibre des classes & Définition des 3 algorithmes
# Les modèles prêts à être entraînés
modeles = {
    'Logistic_Regression': LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42),
    'Naive_Bayes': MultinomialNB(),
    'Random_Forest': RandomForestClassifier(class_weight='balanced', n_estimators=50, random_state=42, n_jobs=-1)
}

# 2. Sauvegarde des modèles vides (construits) pour l'étape d'entraînement
joblib.dump(modeles, 'modeles_construits.pkl')
print("Modèles construits et sauvegardés dans 'modeles_construits.pkl'.")