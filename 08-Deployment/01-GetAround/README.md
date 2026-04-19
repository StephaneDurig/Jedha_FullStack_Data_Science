# GetAround — Projet Deployment (Bloc 5)

![GetAround](https://lever-client-logos.s3.amazonaws.com/2bd4cdf9-37f2-497f-9096-c2793296a75f-1568844229943.png)

## Dépôt source & déploiements live

- **Code source (GitHub)** : [StephaneDurig/Jedha_FullStack_Data_Science — 08-Deployment/01-GetAround](https://github.com/StephaneDurig/Jedha_FullStack_Data_Science/tree/main/08-Deployment/01-GetAround)
- **API de prédiction (Swagger)** : [https://zerphirosx-getaround-api.hf.space/docs](https://zerphirosx-getaround-api.hf.space/docs)
- **Dashboard interactif** : [https://zerphirosx-getaround-dashboard.hf.space](https://zerphirosx-getaround-dashboard.hf.space)
- **Serveur MLflow** : [https://zerphirosx-mlflow.hf.space](https://zerphirosx-mlflow.hf.space)

> Si vous lisez ce fichier depuis un **zip** envoyé par e-mail, le lien GitHub ci-dessus est la
> **source de vérité** : historique de commits, déploiement automatisé via `api/deploy-hf.ps1`,
> structure complète du projet.

> ℹ️ **Note** : le fichier `01-Getaround_analysis.ipynb` à la racine est l'**énoncé original
> fourni par Jedha**, conservé pour traçabilité. Le livrable effectif est
> [`notebooks/01-EDA_and_training.ipynb`](notebooks/01-EDA_and_training.ipynb) — EDA retards,
> entraînement pricing, baselines et enregistrement au Model Registry.

---

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
┌──────────────────────────────────────────────────────────────────────┐
│                         Hugging Face Spaces                          │
│                                                                      │
│  ┌──────────────┐      ┌──────────────┐      ┌──────────────┐        │
│  │  Space 1     │      │  Space 2     │      │  Space 3     │        │
│  │  MLflow      │      │  Dashboard   │      │  FastAPI     │        │
│  │  Server      │      │  Streamlit   │      │  /predict    │        │
│  └───┬──────┬───┘      └──────┬───────┘      └──────┬───────┘        │
│      │      │ artifacts       │ call /predict       │ load model     │
│      ▼      ▼                 ▼                     ▼                │
│   Neon DB   AWS S3 ◄────────────────────── mlflow.pyfunc.load_model  │
│  (backend) (artifacts)            models:/getaround_pricing_model    │
│                                   @production                        │
└──────────────────────────────────────────────────────────────────────┘
                            ▲ entraînement
                            │ log_model → register → set_alias
                      Notebook local
                      (EDA + Training + Registry)
```

| Composant | Technologie | Rôle |
|-----------|-------------|------|
| Notebook | Jupyter + MLflow | EDA retards, entraînement pricing, baselines, Model Registry |
| Dashboard | Streamlit | Analyse interactive des retards + test interactif de l'API pricing |
| API | FastAPI | Endpoint `/predict` pour le pricing (chargement via alias de Registry) |
| MLflow Tracking | Serveur MLflow sur HF | UI des runs, Model Registry, aliases |
| Backend MLflow | Neon DB (PostgreSQL) | Metadata des runs, versions de modèle, aliases |
| Artifact store | AWS S3 | Sérialisation du Pipeline (preprocessor + estimator) |

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
- **Chargement du modèle côté API** — deux options, priorité au Registry :
  1. **Option 1 (recommandée)** : `MLFLOW_REGISTERED_MODEL=getaround_pricing_model` + `MLFLOW_MODEL_ALIAS=production`. L'API charge `models:/<nom>@<alias>` : pour promouvoir une nouvelle version, il suffit de bouger l'alias dans l'UI MLflow, sans redéployer le code.
  2. **Option 2 (fallback)** : `MLFLOW_RUN_ID=<run_id>` du run qui a logué le modèle. Utilisé automatiquement si l'option 1 échoue (backend local `file://`, alias non déclaré, etc.). `MLFLOW_ARTIFACT_PATH` (défaut `getaround_pricing_model`) n'est à modifier que si vous avez changé `artifact_path` dans `mlflow.sklearn.log_model(...)`.
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

**Modèle retenu : Gradient Boosting Regressor**

Choisi pour ses performances sur les données tabulaires hétérogènes. Par rapport à une
régression linéaire, il capture les interactions non-linéaires entre features (ex. une voiture
puissante ET de luxe vaut disproportionnellement plus). Par rapport à Random Forest, il
converge plus vite et obtient de meilleures métriques dans nos tests.

**Baselines de comparaison (section 2.6 du notebook)**

Pour justifier le choix du GBR, deux baselines sont loguées dans la même expérience MLflow
sur le même `train`/`test` split, ce qui rend les `test_rmse`/`test_mae`/`test_r2` directement
comparables dans l'UI MLflow :

| Modèle | Rôle | Attendu |
|--------|------|---------|
| `Ridge` (régression linéaire régularisée) | Plancher simple, purement linéaire | RMSE le plus élevé |
| `RandomForestRegressor` | Baseline non-linéaire, sans boosting | RMSE intermédiaire |
| `GradientBoostingRegressor` | Modèle retenu (boosting séquentiel) | Meilleur RMSE / R² |

Le notebook termine par un `mlflow.search_runs(...)` qui affiche un tableau trié par
`test_rmse` avec les trois modèles côte à côte (preuve chiffrée du choix).

**Pipeline sklearn (ColumnTransformer + GBR)**

Le Pipeline encapsule le preprocessing et le modèle en un seul objet. Cela garantit :
- Pas de data leakage (le scaler est fit uniquement sur le train)
- Reproductibilité parfaite entre entraînement et production
- Sérialisation complète dans MLflow (le modèle logué est directement utilisable)

**Tracking MLflow**

`mlflow.sklearn.autolog()` logue automatiquement tous les hyperparamètres et métriques.
Un `log_model()` explicite avec signature permet à l'API de charger le Pipeline complet
et de faire des prédictions sans preprocessing supplémentaire.

**Model Registry + alias (section 2.7 du notebook)**

Une fois le meilleur run identifié, le notebook appelle `mlflow.register_model(...)` pour créer
une version dans le **MLflow Model Registry** (nom logique : `getaround_pricing_model`), puis
`MlflowClient().set_registered_model_alias(name, alias, version)` pour poser deux aliases :

| Alias | Rôle |
|-------|------|
| `champion` | Meilleure version sur le dataset courant (doit rester stable) |
| `production` | Version réellement servie par l'API en prod |

L'API charge ensuite **`models:/getaround_pricing_model@production`** plutôt qu'un
`runs:/<run_id>/...` figé. **Promouvoir une nouvelle version** = changer l'alias
`production` dans l'UI MLflow ; le Space **getaround-api** n'a rien à redéployer.

**Backend MLflow : Neon DB (Postgres) + S3**

Le serveur MLflow sur Hugging Face utilise **Neon DB** (PostgreSQL serverless gratuit)
comme *backend store* (metadata des runs, versions, aliases) et **AWS S3** comme *artifact
store* (Pipeline sérialisé). Ce couple Postgres + S3 est **requis** pour activer le Model
Registry — un backend `file://` ne supporte ni les versions ni les aliases.

**Version du client MLflow (tracking distant HF)**

Le serveur MLflow sur Hugging Face est en **2.21.x**. Utiliser le **même** client dans le
notebook et l'API pour éviter des incompatibilités de schéma :

`pip install mlflow==2.21.3`

(voir `api/requirements.txt` et `notebooks/requirements-training.txt` si présent.)

---

## API FastAPI — Endpoint `/predict`

L'API attend un JSON avec la clé **`input`** : liste d'objets véhicule (mode *batch* supporté).
Chaque véhicule est un **dictionnaire typé** validé par Pydantic (avec `Literal[...]` sur les
champs catégoriels) — l'ordre des clés n'a pas d'importance, mais **toutes** sont obligatoires.
Schéma complet sous `/docs` (Swagger) : [https://zerphirosx-getaround-api.hf.space/docs](https://zerphirosx-getaround-api.hf.space/docs).

La réponse est `{"prediction": [<prix en €/jour>, ...]}` (une valeur par véhicule).

### Utilisation (curl)

```bash
curl -X POST "https://zerphirosx-getaround-api.hf.space/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "input": [
      {
        "model_key": "Citroën",
        "fuel": "diesel",
        "paint_color": "black",
        "car_type": "hatchback",
        "mileage": 140000,
        "engine_power": 90,
        "private_parking_available": true,
        "has_gps": true,
        "has_air_conditioning": true,
        "automatic_car": false,
        "has_getaround_connect": true,
        "has_speed_regulator": true,
        "winter_tires": false
      }
    ]
  }'
```

### En Python

```python
import requests

url = "https://zerphirosx-getaround-api.hf.space/predict"
payload = {
    "input": [
        {
            "model_key": "Citroën",
            "fuel": "diesel",
            "paint_color": "black",
            "car_type": "hatchback",
            "mileage": 140000,
            "engine_power": 90,
            "private_parking_available": True,
            "has_gps": True,
            "has_air_conditioning": True,
            "automatic_car": False,
            "has_getaround_connect": True,
            "has_speed_regulator": True,
            "winter_tires": False,
        }
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

> Un payload au format historique (liste de 13 valeurs positionnelles) renverra désormais une
> erreur **422 Unprocessable Entity** : l'API a été durcie avec un schéma Pydantic explicite
> pour éviter toute ambiguïté sur l'ordre des features et améliorer le retour d'erreur.

---

## Installation locale

### Prérequis

- Python 3.9+ (3.12 recommandé pour coller à l’environnement d’entraînement / Docker API)
- Conda (recommandé)
- Compte Hugging Face
- Compte AWS avec accès au bucket S3 utilisé par MLflow (pour charger le modèle côté API)

### Setup

```bash
# Cloner le repository puis se placer dans le sous-dossier du projet
git clone https://github.com/StephaneDurig/Jedha_FullStack_Data_Science.git
cd Jedha_FullStack_Data_Science/08-Deployment/01-GetAround

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

# Linux / macOS — Option 1 : charger le modèle via le Model Registry (recommandé)
export MLFLOW_TRACKING_URI="https://zerphirosx-mlflow.hf.space"
export MLFLOW_REGISTERED_MODEL="getaround_pricing_model"
export MLFLOW_MODEL_ALIAS="production"
# export MLFLOW_ARTIFACT_PATH="getaround_pricing_model"   # optionnel (valeur par défaut)
export AWS_ACCESS_KEY_ID="..."
export AWS_SECRET_ACCESS_KEY="..."

# Option 2 (fallback) : si le Registry n'est pas utilisé, fournir le Run ID à la place
# export MLFLOW_RUN_ID="votre-run-id"

# Windows (PowerShell)
# $env:MLFLOW_TRACKING_URI="https://zerphirosx-mlflow.hf.space"
# $env:MLFLOW_REGISTERED_MODEL="getaround_pricing_model"
# $env:MLFLOW_MODEL_ALIAS="production"

fastapi run app.py --port 7860
# Documentation : http://localhost:7860/docs
```

L'API charge en priorité `models:/$MLFLOW_REGISTERED_MODEL@$MLFLOW_MODEL_ALIAS` ; si cet
appel échoue (backend file://, alias inconnu, credentials manquants), elle retombe sur
`runs:/$MLFLOW_RUN_ID/$MLFLOW_ARTIFACT_PATH`. Un message explicite est affiché au démarrage
pour indiquer quelle source a été utilisée.

---

## Déploiement pas à pas

Enchaînement type : configurer les **secrets** et variables sur Hugging Face (Spaces MLflow, API, dashboard), renseigner `.env` à partir de [`.env.example`](.env.example), entraîner / enregistrer le modèle avec le notebook (`notebooks/`) si besoin, puis déployer et tester les URLs listées dans la section **URLs de production** ci-dessus.

---

## Auteur

Projet réalisé dans le cadre du **Bloc 5 — Deployment** de la formation
**Data Science Full Stack** à [Jedha Bootcamp](https://jedha.co).
