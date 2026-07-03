# Détection Automatique de Tweets Suspects avec Machine Learning

## Description

L'objectif est de développer un système capable de classifier automatiquement un tweet en deux catégories :

- **Classe 0 : Tweet suspect**
- **Classe 1 : Tweet non suspect**

Le projet couvre l'ensemble du cycle de vie d'un projet de Machine Learning, depuis le prétraitement des données jusqu'au déploiement d'une interface utilisateur, en utilisant **Git** pour le versionnement du code, **DVC (Data Version Control)** pour la gestion des données et du pipeline, ainsi que **uv** pour la gestion de l'environnement Python.

---

# Technologies utilisées

- Python
- Scikit-learn
- Pandas
- NLTK
- Streamlit
- Git
- DVC
- uv

---

# Structure du pipeline DVC

Le pipeline est défini dans le fichier `dvc.yaml` et comprend les étapes suivantes :

```
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
```

---

# Installation

## 1. Cloner le projet

```bash
git clone https://github.com/ouattarafaiza/Detection_tweets_suspect.git
cd projet_tweets_suspect
```

---

## 2. Installer les dépendances avec uv

Le projet utilise **uv** comme gestionnaire d'environnement et de dépendances.

Installer les packages :

```bash
uv pip install dvc scikit-learn pandas numpy matplotlib seaborn joblib streamlit
```


---

## 3. Reproduire entièrement le pipeline

Le pipeline complet peut être reconstruit automatiquement grâce à DVC :

```bash
dvc repro
```

Cette commande exécute automatiquement les étapes suivantes :

- Prétraitement
- Vectorisation TF-IDF
- Construction des modèles
- Entraînement
- Validation
- Génération des graphiques
- Optimisation du modèle
- Préparation du déploiement

---

# Lancer l'application

L'application de démonstration est développée avec **Streamlit**.

Exécuter :

```bash
uv run streamlit run app.py
```

Puis ouvrir le navigateur à l'adresse :

```
http://localhost:8501
```

---

# Modèles évalués

Les modèles comparés sont :

- Régression Logistique
- Naive Bayes
- Random Forest

Le modèle retenu est **Random Forest**, sélectionné après comparaison des performances et optimisation par **Grid Search**.

---

# Gestion du déséquilibre

Le jeu de données présente un fort déséquilibre entre les classes.

Pour limiter ce biais, la stratégie **class_weight="balanced"** a été utilisée lors de l'entraînement des modèles.

---

# Représentation des données

Les tweets sont représentés sous forme de vecteurs numériques grâce à **TF-IDF (Term Frequency – Inverse Document Frequency)**.

Cette méthode permet de mieux valoriser les termes discriminants tout en réduisant l'influence des mots très fréquents.

---

# Reproductibilité

Le projet est entièrement reproductible grâce à :

- **Git** : versionnement du code source.
- **DVC** : versionnement des données et pipeline.
- **uv** : gestion rapide et reproductible de l'environnement Python.

# Stockage distant DVC

Le projet utilise un **stockage distant local** configuré avec DVC afin de versionner les données et les artefacts générés par le pipeline.

Configuration utilisée :

```bash
dvc remote add -d my_remote /tmp/dvc_remote
```

Les fichiers suivis par DVC peuvent être envoyés vers le stockage distant avec :

```bash
dvc push
```

Ils peuvent être récupérés sur une autre machine à l'aide de :

```bash
dvc pull
```


Les principales commandes sont :

```bash
uv pip install dvc scikit-learn pandas numpy matplotlib seaborn joblib streamlit

dvc pull

dvc repro

uv run streamlit run app.py
```

---
