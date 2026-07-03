import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import os

# Téléchargement des dépendances NLTK
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)

def pipeline_pretraitement(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'@\w+|#\w+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    words = text.split()
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    cleaned_words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    return " ".join(cleaned_words)

if __name__ == "__main__":
    print("Début du prétraitement...")
    # Lecture du fichier d'entrée
    df = pd.read_csv('tweets_suspect.csv')
    
    # Traitement
    df['clean_message'] = df['message'].apply(pipeline_pretraitement)
    
    # Sauvegarde du fichier de sortie
    df.to_csv('tweets_suspect_clean.csv', index=False)
    print("Prétraitement terminé. Fichier 'tweets_suspect_clean.csv' créé.")