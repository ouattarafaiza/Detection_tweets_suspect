# src_vectorisation.py
import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer

print("--- Début de l'étape : Représentation des données (TF-IDF) ---")

# 1. Chargement des données nettoyées générées par l'étape précédente
df = pd.read_csv('tweets_suspect_clean.csv')

# Gestion des valeurs manquantes qui auraient pu apparaître si un tweet ne contenait que des stop words
df['clean_message'] = df['clean_message'].fillna('')

X_text = df['clean_message']
y = df['label']

# 2. Initialisation et ajustement du modèle TF-IDF
# On limite à 5000 fonctionnalités (mots clés) pour éviter d'exploser la mémoire vive
print("Ajustement du vectoriseur TF-IDF sur les tweets...")
tfidf = TfidfVectorizer(max_features=5000)
X_tfidf = tfidf.fit_transform(X_text)

# 3. Sauvegarde de la matrice et du vectoriseur)
print("Sauvegarde de la matrice et du vectoriseur...")
joblib.dump((X_tfidf, y), 'matrix_tfidf.pkl')
joblib.dump(tfidf, 'tfidf_vectorizer.pkl')

print("Fichiers 'matrix_tfidf.pkl' et 'tfidf_vectorizer.pkl' créés.")