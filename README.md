# Détection Automatique de Tweets Suspects avec Machine Learning

## Description

Ce projet a pour objectif de développer un système de classification automatique de tweets en deux catégories :

Classe 0 : Tweet suspect
Classe 1 : Tweet non suspect

Le projet couvre l’ensemble du cycle de vie d’un projet de Machine Learning :

Exploration des données
Prétraitement du texte
Vectorisation
Modélisation
Évaluation
Optimisation
Déploiement

Il utilise :

Git pour le versionnement du code
DVC (Data Version Control) pour les données et le pipeline
uv pour la gestion de l’environnement Python

### Technologies utilisées
Python
Scikit-learn
Pandas
NumPy
NLTK
Matplotlib / Seaborn
Streamlit
Git
DVC
uv

### Pipeline DVC

Le pipeline est défini dans dvc.yaml et comprend les étapes suivantes :

tweets_suspect.csv
        │
        ▼
Prétraitement
(src_pretraitement.py)
        │
        ▼
Vectorisation TF-IDF
(src_vectorisation.py)
        │
        ▼
Construction des modèles
(src_modeles.py)
        │
        ▼
Entraînement et validation
(src_entrainement_validation.py)
        │
        ▼
Évaluation graphique
(src_evaluation_graphique.py)
        │
        ▼
Optimisation (Grid Search)
(src_optimisation.py)
        │
        ▼
Application Streamlit
(app.py)


###  Installation

1️. Cloner le projet

git clone https://github.com/ouattarafaiza/Detection_tweets_suspect.git
cd Detection_tweets_suspect

2️. Installer les dépendances,  exécutez : 


uv pip install dvc scikit-learn pandas numpy matplotlib seaborn joblib streamlit nltk

### Reproductibilité du projet (IMPORTANT)

Le projet est entièrement reproductible grâce à Git + DVC.

- Stockage distant DVC (LOCAL REMOTE)

Le projet utilise un remote DVC local qui est fourni atravers le dossier nommer DVC_Remote_Tweets qui doit etre placer dans le disque local C.

- Configuration du remote

Après clonage du projet, configurer le remote en exécutant  :

dvc remote add -d myremote C:/DVC_Remote_Tweets
dvc remote default myremote

- Récupération des données , exécutez: 

dvc pull

Cette commande permet de récupérer :

dataset original
fichiers intermédiaires
modèles entraînés
résultats d’évaluation

- Reproduction complète du pipeline, exécutez: 

dvc repro

Cette commande exécute automatiquement :

Prétraitement des données
Vectorisation TF-IDF
Construction des modèles
Entraînement et validation
Évaluation
Optimisation
Préparation du modèle final


#### Lancer l’application
Pour tester le modèle final optimisé en temps réel à l'aide d'une interface graphique interactive, exécutez :

uv run streamlit run app.py

Puis ouvrir :

http://localhost:8501

### Proposition tweets pour test

- Beautiful day today, just finished my data science project! Have a great weekend everyone.

- Happy birthday to my best friend! Wishing you an amazing year filled with joy, success, and happiness.

- Just checking the weather for tomorrow, looks like it's going to be sunny all day. Perfect for a walk.

- I hate you so much, I will find you and make you pay for this.

- You are absolutely useless and a piece of trash. Get off the internet, nobody likes you.

- Shut up you absolute idiot, your opinion is completely trash and you have zero brain cells.


