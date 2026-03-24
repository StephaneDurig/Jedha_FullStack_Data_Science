# Uber Pickups - Identification des Hot-Zones à New York City

Projet de Machine Learning Non-Supervisé | Bloc 3 - JEDHA Data Science Fullstack

---

## Contexte et problématique

**Uber** fait face à un problème récurrent : les chauffeurs ne sont pas toujours positionnés dans les zones à forte demande, ce qui entraîne des temps d'attente excessifs (>7 min) et des annulations de courses.

**Objectif** : Identifier les **hot-zones** (zones à forte concentration de pickups) à New York City pour recommander aux chauffeurs les meilleurs emplacements en fonction du jour de la semaine.

## Données

- **Source** : [Uber Trip Data](https://full-stack-bigdata-datasets.s3.eu-west-3.amazonaws.com/Machine+Learning+non+Supervis%C3%A9/Projects/uber-trip-data.zip)
- **Périmètre** : Pickups Uber à New York City, avril-septembre 2014
- **Colonnes** : `Date/Time`, `Lat`, `Lon`, `Base`
- **Volume** : ~4.5 millions de courses (fichiers `uber-raw-data-*.csv` uniquement ; le fichier auxiliaire `taxi-zone-lookup.csv` n'est pas fusionné aux courses)

## Méthodologie

### 1. Analyse Exploratoire (EDA)

- Distributions temporelles (heure, jour, mois, base)
- Heatmap jour x heure
- Carte des pickups bruts (Plotly Mapbox)

### 2. Pipeline de préparation

- Filtrage géographique (bounding box NYC)
- Suppression des doublons
- Feature engineering temporel (heure, jour de la semaine, mois)
- Normalisation avec `StandardScaler`

### 3. Réduction de dimensionnalité (PCA)

- Application de la PCA sur les features étendues (Lat, Lon, heure, jour)
- Lecture prudente de la variance expliquée pour appuyer l'idée de structure géographique dominante

### 4. Clustering K-Means

- **Approche incrémentale** : début sur un sous-ensemble (Lundi 17h), puis généralisation
- **Optimisation** : méthode du coude (aide visuelle) + score silhouette, avec contrainte **K ≥ 5** puis choix du K qui maximise la silhouette
- **Amélioration itérative** : comparaison V1 (K arbitraire) vs V2 (K ainsi choisi)
- **Visualisation** : cartes interactives avec centroïdes (hot-zones)

### 5. Clustering DBSCAN

- **Aide à eps** : courbe des k-plus proches voisins (ordre de grandeur)
- **Grille** : exploration systématique de `eps` et `min_samples`
- **Amélioration itérative** : V1 (paramètres par défaut) vs V2 (optimisé)
- **Gestion du bruit** : identification des points isolés

### 6. Comparaison des algorithmes

- Tableau comparatif (Silhouette Score, nombre de clusters, gestion du bruit)
- Cartes côte à côte
- Discussion des forces et faiblesses de chaque approche

### 7. Généralisation

- Clustering par jour de la semaine (même K que sur Lundi 17h, toutes heures confondues)
- Analyse des patterns semaine vs weekend
- Carte synthétique de tous les centroïdes

## Choix techniques et justifications

| Choix | Justification |
|-------|---------------|
| **Apprentissage non-supervisé** | Pas de variable cible, objectif de découverte de structure |
| **K-Means** | Centroïdes = coordonnées GPS exploitables pour les chauffeurs |
| **DBSCAN** | Détection du bruit et formes arbitraires, complémentaire à K-Means |
| **PCA** | Explorer la variance et la structure avant le clustering |
| **Plotly Mapbox** | Cartes interactives professionnelles |
| **Score Silhouette** | Mesure objective de la qualité du clustering |
| **StandardScaler** | Normalisation nécessaire pour les distances euclidiennes |

## Structure du projet

```
01-Uber_Pickups/
├── 01-Uber_Pickups.ipynb              # Énoncé du projet
├── 01-Uber_Pickups_solutions.ipynb    # Analyse complète (corrigée)
├── README.md                          # Ce fichier
└── uber-trip-data/                    # Données CSV (ou téléchargement via le notebook)
```

## Technologies utilisées

- **Python 3.10+**
- **pandas** / **numpy** : manipulation et calcul numérique
- **scikit-learn** : `KMeans`, `DBSCAN`, `StandardScaler`, `PCA`, `silhouette_score`, `NearestNeighbors`
- **plotly** : cartes interactives et visualisations (Express + Graph Objects)
- **matplotlib** : visualisations complémentaires

## Exécution

1. Ouvrir le notebook `01-Uber_Pickups_solutions.ipynb` dans Jupyter
2. Exécuter toutes les cellules séquentiellement (le téléchargement des données est automatique)
3. Les cartes interactives s'affichent directement dans le notebook

**Prérequis** :
```bash
pip install pandas numpy scikit-learn plotly matplotlib
```

## Résultats clés

- **Hot-zones stables** : les zones à forte densité de pickups sont concentrées à Manhattan (Midtown, Financial District, Upper East/West Side)
- **Patterns temporels** : la demande en semaine est supérieure au weekend, avec des hot-zones qui se déplacent légèrement
- **K-Means** fournit des centroïdes directement exploitables comme recommandations GPS
- **DBSCAN** complète l'analyse en identifiant les zones de bruit à éviter
- **Amélioration mesurable** : les réglages optimisés surpassent les versions naïves sur le score de silhouette
