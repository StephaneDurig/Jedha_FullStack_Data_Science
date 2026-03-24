# Plan Your Trip with Kayak (Bloc 1 — Data Collection and Management)

Projet **Data Collection and Management** : pipeline complète de la collecte (APIs, scraping) au **data lake** (S3) et à l’**entrepôt de données** (PostgreSQL sur **Neon**), avec visualisations interactives.

## Synthèse du projet

Kayak souhaite recommander des destinations en France sur la base de données **réelles** : météo à 7 jours et offre hôtelière. Nous couvrons **35 villes** imposées par le sujet, calculons un **score météo** interprétable, collectons des **hôtels sur Booking.com** pour **chaque** ville, stockons les exports CSV dans **Amazon S3** sous le préfixe du lab (`Bloc-1/Kayak/raw/`), puis chargeons les données nettoyées dans **Neon** pour que les équipes SQL puissent les exploiter. Les cartes Plotly présentent le **Top-5** des destinations et le **Top-20** des hôtels.

## Architecture

```
Nominatim + OpenWeatherMap + Booking.com
        → Python (requests, Selenium, pandas)
        → S3  s3://cloudlab-certification-b6bff020/Bloc-1/Kayak/raw/*.csv
        → ETL (pandas) → Neon PostgreSQL (tables cities, weather, hotels)
        → Plotly (cartes et graphiques)
```

- **Data lake** : fichiers CSV bruts / enrichis sur S3.
- **Data warehouse** : base relationnelle Neon (équivalent au « RDS » du sujet initial ; hébergement managé PostgreSQL avec `NEON_DATABASE_URL`).

## Livrables

| Livrable | Détail |
|----------|--------|
| Notebook de référence | [`01-Plan_your_trip_with_Kayak_solutions.ipynb`](01-Plan_your_trip_with_Kayak_solutions.ipynb) |
| CSV sur S3 | `cities.csv`, `weather_data.csv`, `city_weather_scores.csv`, `hotels_data.csv`, **`enriched_kayak.csv`** (jointure hôtels + scores météo par ville) |
| Base SQL | Tables `cities`, `weather`, `hotels` dans Neon |
| Cartes | Top-5 destinations météo, Top-20 hôtels (scores utilisateur) |

## Prérequis

- Python 3.10+
- Google Chrome (scraping Selenium)
- Compte **OpenWeatherMap** (One Call 3.0)
- Compte **AWS** avec utilisateur IAM limité au **S3** du bucket lab
- Projet **Neon** (PostgreSQL) et connection string
- Fichier **`.env`** (voir [`.env.example`](.env.example))

## Configuration

1. Copier `.env.example` vers `.env`.
2. Renseigner `OPENWEATHERMAP_API_KEY`, `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `NEON_DATABASE_URL`.
3. Vérifier `S3_BUCKET_NAME` et `S3_PREFIX` (valeurs par défaut alignées sur le lab : `cloudlab-certification-b6bff020`, `Bloc-1/Kayak`).

La variable **`NEON_DATABASE_URL`** doit contenir la chaîne complète fournie par Neon (avec `sslmode=require`). Le notebook adapte le préfixe pour SQLAlchemy (`postgresql+psycopg2`).

## Score météo (rappel)

Pondération indicative : température 40 %, probabilité de précipitation 30 %, humidité 15 %, vent 15 %. Le détail est dans le notebook ; le champ pluie API est **normalisé** (nombre ou dictionnaire) avant agrégation.

## Conformité RGPD (rappel)

Pas de collecte de données personnelles dans le périmètre métier choisi ; rate limiting sur les sources ; secrets hors dépôt Git ; accès AWS limité au bucket ; connexion Neon chiffrée (TLS).

## Foire aux questions (FAQ)

- **Pourquoi Neon plutôt que RDS ?** Même rôle (PostgreSQL pour l’ETL), environnement imposé par le lab ; SQLAlchemy inchangé.
- **Pourquoi plusieurs CSV + un enrichi ?** Normalisation en tables + fichier unique `enriched_kayak.csv` pour le livrable « météo + hôtels ».
- **Scraping fragile ?** Booking change souvent le HTML ; des délais et un périmètre 35 villes augmentent le temps d’exécution — prévoir une exécution longue ou un environnement stable.

## Cahier des charges

- Énoncé du projet : [`01-Plan_your_trip_with_Kayak.ipynb`](01-Plan_your_trip_with_Kayak.ipynb)

## Technologies

Python, pandas, requests, boto3, SQLAlchemy, psycopg2-binary, Selenium, BeautifulSoup, Plotly, python-dotenv.
