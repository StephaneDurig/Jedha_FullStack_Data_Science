# GetAround — Projet Deployment (Bloc 5)

![GetAround](https://lever-client-logos.s3.amazonaws.com/2bd4cdf9-37f2-497f-9096-c2793296a75f-1568844229943.png)

## Présentation du projet

GetAround est une plateforme de location de voitures entre particuliers (le "Airbnb des voitures").
Ce projet aborde deux problématiques métier distinctes :

1. **Analyse des retards** : aider le Product Manager à définir un délai minimum entre deux locations
   pour réduire les frictions liées aux retards au checkout.
2. **Optimisation des prix** : proposer aux propriétaires un prix journalier optimal via un modèle
   de Machine Learning, accessible via une API en production.

---

## Architecture du projet

```
┌─────────────────────────────────────────────────────────────┐
│                    Hugging Face Spaces                      │
│                                                             │
│  ┌──────────────┐   ┌──────────────┐   ┌──────────────┐     │
│  │  Space 1     │   │  Space 2     │   │  Space 3     │     │
│  │  MLflow      │   │  Dashboard   │   │  FastAPI     │     │
│  │  Server      │   │  Streamlit   │   │  /predict    │     │
│  └──────┬───────┘   └──────────────┘   └──────┬───────┘     │
│         │ artifacts                            │ load model │
│         ▼                                      ▼            │
│     AWS S3 ◄──────────────────────────── mlflow.pyfunc      │
└─────────────────────────────────────────────────────────────┘
                            ▲ entraînement
                            │ (run_id → API)
                      Notebook local
                      (EDA + Training)
```

| Composant | Technologie | Rôle |
|-----------|-------------|------|
| Notebook | Jupyter + MLflow | EDA, entraînement, tracking |
| Dashboard | Streamlit | Analyse interactive des retards + test API pricing |
| API | FastAPI | Endpoint `/predict` pour le pricing |
| Model store | MLflow + S3 | Versioning et stockage du modèle |

---

## URLs de production (déploiement ZerphirosX)

| Service | URL |
|---------|-----|
| Serveur MLflow | https://zerphirosx-mlflow.hf.space |
| Dashboard Streamlit | https://zerphirosx-getaround-dashboard.hf.space |
| API — prédiction | https://zerphirosx-getaround-api.hf.space/predict |
| API — documentation | https://zerphirosx-getaround-api.hf.space/docs |

> Pour un autre compte Hugging Face, remplacer le préfixe `zerphirosx` par votre identifiant.

---

## Secrets, `.env` et dépôt Git

- **Ne jamais committer** le fichier `.env` (il est listé dans `.gitignore`).
- **Clés AWS** : uniquement dans les **Secrets / Variables** Hugging Face (Spaces API et MLflow selon le cas) et dans votre `.env` **local**. Aucune clé réelle ne doit figurer dans le code ou le README.
- Copier `.env.example` vers `.env` et renseigner les valeurs (voir commentaires dans `.env.example`).
- **`MLFLOW_RUN_ID`** : doit être **le même** Run ID que celui du modèle enregistré sur le serveur MLflow (HF), à la fois dans `.env` local et dans les secrets du Space **getaround-api**.
- **`GETAROUND_API_URL`** : URL de base de l’API **sans** `/predict` (ex. `https://zerphirosx-getaround-api.hf.space`). À définir dans `.env` pour le développement local du dashboard et en **variable d’environnement** du Space **getaround-dashboard** sur HF pour préremplir le formulaire de test.

---

## Données

| Fichier | Source | Usage |
|---------|--------|-------|
| `get_around_delay_analysis.xlsx` | Jedha (S3 public ou `src/`) | Analyse des retards (notebook, dashboard) |
| `get_around_pricing_project.csv` | Jedha (fichier dans `src/` ou URL Jedha) | Entraînement du modèle ML |

Liens publics Jedha (même contenu que `src/`) :

- Delay : `https://full-stack-assets.s3.eu-west-3.amazonaws.com/Deployment/get_around_delay_analysis.xlsx`
- Pricing : `https://full-stack-assets.s3.eu-west-3.amazonaws.com/Deployment/get_around_pricing_project.csv`

---

## Analyse des retards — Résultats clés

Le Product Manager doit décider d'un **threshold** (délai minimum entre deux locations)
et d'un **scope** (toutes les voitures ou seulement Connect).

**Principaux résultats :**

- ~45% des locations se terminent avec un retard au checkout
- Les locations Connect ont un profil de retard différent des Mobile
- Les retards dépassent 60 min dans environ 20% des cas tardifs
- Des dizaines à centaines de locations consécutives sont impactées chaque période

**Recommandation :**
Un threshold de **60 à 120 minutes** avec scope **Connect uniquement** offre le meilleur
compromis entre réduction des frictions et préservation des revenus.

Consulter le dashboard interactif pour simuler n'importe quelle combinaison.

---

## Modèle ML — Pricing

### Données

Le dataset `get_around_pricing_project.csv` contient ~4800 voitures avec leurs caractéristiques
et leur prix journalier de location.

**Features :**

| Feature | Type | Description |
|---------|------|-------------|
| `model_key` | Catégorielle | Marque du véhicule |
| `mileage` | Numérique | Kilométrage |
| `engine_power` | Numérique | Puissance (chevaux) |
| `fuel` | Catégorielle | Type de carburant |
| `paint_color` | Catégorielle | Couleur |
| `car_type` | Catégorielle | Type de véhicule |
| `private_parking_available` | Booléenne | Parking privé |
| `has_gps` | Booléenne | GPS |
| `has_air_conditioning` | Booléenne | Climatisation |
| `automatic_car` | Booléenne | Boîte automatique |
| `has_getaround_connect` | Booléenne | Connect |
| `has_speed_regulator` | Booléenne | Régulateur |
| `winter_tires` | Booléenne | Pneus hiver |

**Cible :** `rental_price_per_day` (prix en €/jour)

### Choix techniques

**Modèle : Gradient Boosting Regressor**

Choisi pour ses performances sur les données tabulaires hétérogènes. Par rapport à une
régression linéaire, il capture les interactions non-linéaires entre features (ex. une voiture
puissante ET de luxe vaut disproportionnellement plus). Par rapport à Random Forest, il
converge plus vite et obtient de meilleures métriques dans nos tests.

**Pipeline sklearn (ColumnTransformer + GBR)**

Le Pipeline encapsule le preprocessing et le modèle en un seul objet. Cela garantit :
- Pas de data leakage (le scaler est fit uniquement sur le train)
- Reproductibilité parfaite entre entraînement et production
- Sérialisation complète dans MLflow (le modèle logué est directement utilisable)

**Tracking MLflow**

`mlflow.sklearn.autolog()` logue automatiquement tous les hyperparamètres et métriques.
Un `log_model()` explicite avec signature permet à l'API de charger le Pipeline complet
et de faire des prédictions sans preprocessing supplémentaire.

**Version du client MLflow (tracking distant HF)**

Le serveur MLflow sur Hugging Face est en **2.21.x**. Utiliser le **même** client dans le notebook, par exemple :

`pip install mlflow==2.21.3`

(voir aussi `notebooks/requirements-training.txt` si présent.)

---

## API FastAPI — Endpoint `/predict`

L’API attend un JSON avec la clé **`input`** : liste de lignes ; chaque ligne est une liste de
**13 valeurs** dans l’ordre du notebook [`notebooks/01-EDA_and_training.ipynb`](notebooks/01-EDA_and_training.ipynb)
: `model_key`, `fuel`, `paint_color`, `car_type`, `mileage`, `engine_power`, puis les sept
booléens (`private_parking_available` … `winter_tires`). La réponse est
`{"prediction": [<prix en €/jour>, ...]}` (une valeur par ligne).

### Utilisation (curl)

```bash
curl -X POST "https://zerphirosx-getaround-api.hf.space/predict" \
  -H "Content-Type: application/json" \
  -d "{\"input\":[[\"Citroën C3\",\"diesel\",\"black\",\"hatchback\",140000,90,1,1,1,0,1,1,0]]}"
```

### En Python

```python
import requests

url = "https://zerphirosx-getaround-api.hf.space/predict"
payload = {
    "input": [
        [
            "Citroën C3",
            "diesel",
            "black",
            "hatchback",
            140000,
            90,
            1,
            1,
            1,
            0,
            1,
            1,
            0,
        ]
    ],
}
response = requests.post(url, json=payload, timeout=60)
print(response.json())
```

**Réponse (exemple) :**

```json
{
  "prediction": [99.76]
}
```

---

## Installation locale

### Prérequis

- Python 3.9+ (3.12 recommandé pour coller à l’environnement d’entraînement / Docker API)
- Conda (recommandé)
- Compte Hugging Face
- Compte AWS avec accès au bucket S3 utilisé par MLflow (pour charger le modèle côté API)

### Setup

```bash
# Cloner le repository
git clone https://github.com/VOTRE_USERNAME/getaround-deployment
cd getaround-deployment

# Créer l'environnement
conda create -n getaround python=3.12 -y
conda activate getaround

# Installer les dépendances pour le notebook (ajuster selon requirements du cours)
pip install pandas numpy plotly openpyxl "mlflow==2.21.3" scikit-learn jupyter

# Lancer Jupyter
jupyter notebook notebooks/01-EDA_and_training.ipynb
```

### Lancer le dashboard localement

```bash
cd dashboard/
pip install -r requirements.txt
streamlit run app.py
# → http://localhost:8501
```

Définir au besoin `GETAROUND_API_URL` dans `.env` à la racine du projet (voir `.env.example`).

### Lancer l'API localement

```bash
cd api/
pip install -r requirements.txt

# Linux / macOS — aligner avec votre store MLflow et le run du modèle
export MLFLOW_TRACKING_URI="https://zerphirosx-mlflow.hf.space"
export MLFLOW_RUN_ID="votre-run-id"
export AWS_ACCESS_KEY_ID="..."
export AWS_SECRET_ACCESS_KEY="..."

# Windows (PowerShell)
# $env:MLFLOW_TRACKING_URI="https://zerphirosx-mlflow.hf.space"
# $env:MLFLOW_RUN_ID="votre-run-id"

fastapi run app.py --port 7860
# Documentation : http://localhost:7860/docs
```

---

## Déploiement pas à pas

Enchaînement type : configurer les **secrets** et variables sur Hugging Face (Spaces MLflow, API, dashboard), renseigner `.env` à partir de [`.env.example`](.env.example), entraîner / enregistrer le modèle avec le notebook (`notebooks/`) si besoin, puis déployer et tester les URLs listées dans la section **URLs de production** ci-dessus.

---

## Auteur

Projet réalisé dans le cadre du **Bloc 5 — Deployment** de la formation
**Data Science Full Stack** à [Jedha Bootcamp](https://jedha.co).
