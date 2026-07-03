# app.py
import streamlit as st
import joblib

# 1. Configuration de la page Streamlit
st.set_page_config(page_title="Détecteur de Tweets Suspects", page_icon="🛡️", layout="centered")

st.title("🛡️ Analyseur de Tweets - Détection de Suspicions")
st.write("Cette interface permet d'évaluer en temps réel si un tweet présente un caractère suspect ou malveillant.")

# 2. Chargement des artefacts du modèle (Vectoriseur et Modèle optimisé)
@st.cache_resource # Permet de ne charger le modèle qu'une seule fois en mémoire
def charger_modeles():
    vectorizer = joblib.load('tfidf_vectorizer.pkl')
    model = joblib.load('modele_final_optimise.pkl')
    return vectorizer, model

try:
    tfidf, modele_final = charger_modeles()
    st.success("Modèle pré-entraîné chargé avec succès")
except Exception as e:
    st.error(f"Erreur lors du chargement des modèles : {e}")

st.write("---")

# 3. Zone de saisie utilisateur
user_tweet = st.text_area("Saisissez le texte du tweet à analyser :", placeholder="Écrivez votre tweet ici...", height=100)

if st.button("Analyser le tweet"):
    if user_tweet.strip() == "":
        st.warning("Veuillez saisir un texte avant de lancer l'analyse.")
    else:
        # 4. Transformation TF-IDF et Prédiction
        # On passe le texte dans une liste car le vectoriseur attend un itérable de documents
        tweet_vectorise = tfidf.transform([user_tweet])
        
        prediction = modele_final.predict(tweet_vectorise)[0]
        probabilites = modele_final.predict_proba(tweet_vectorise)[0]
        
        # Probabilité de la classe 1 (Suspect)
        probabilite_suspect = probabilites[1] * 100
        
        st.write("### Résultats de l'analyse :")
        
        # 5. Affichage dynamique corrigé selon tes labels : 0 = Suspect, 1 = Non-Suspect
        if prediction == 0:
            st.error(f"🚨 **Résultat : Tweet Suspect (Offensif / Critique)**")
            st.write(f"Le modèle estime à **{(probabilites[0] * 100):.2f}%** la probabilité que ce contenu soit suspect.")
            st.progress(probabilites[0]) # Barre de progression basée sur le risque (classe 0)
        else:
            st.success(f"✅ **Résultat : Tweet Non-Suspect (Normal / Positif)**")
            st.write(f"Le modèle estime à **{(probabilites[1] * 100):.2f}%** la probabilité que ce contenu soit sain.")
            st.progress(probabilites[0]) # La barre montre quand même le niveau de risque résiduel