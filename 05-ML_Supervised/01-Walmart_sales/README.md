# Walmart : Prédiction des ventes hebdomadaires

## Description du projet

Le service marketing de **Walmart** souhaite disposer d'un modèle de Machine Learning capable
d'**estimer les ventes hebdomadaires** de ses magasins. L'objectif est double :

- Comprendre l'influence des indicateurs économiques sur les ventes.
- Fournir un outil de prédiction fiable pour planifier les futures campagnes marketing.

Ce projet s'inscrit dans le cadre du **Bloc 3** de la certification Data Science et répond
aux critères de la grille d'évaluation associée.

**Fichiers notebook :**

- [`01-Walmart_sales.ipynb`](01-Walmart_sales.ipynb) — **énoncé** du projet (consignes et objectifs).
- [`01-Walmart_sales_solution.ipynb`](01-Walmart_sales_solution.ipynb) — **corrigé détaillé** (EDA, preprocessing, modèles, figures et tableaux de résultats).

## Dataset

| Fichier | Description |
|---------|-------------|
| `Walmart_Store_sales.csv` | Données hebdomadaires de ventes Walmart |

**Variables :**

| Colonne | Type | Description |
|---------|------|-------------|
| `Store` | Catégoriel | Identifiant du magasin |
| `Date` | Date | Date de la semaine (DD-MM-YYYY) |
| `Weekly_Sales` | Numérique | **Variable cible** — Ventes hebdomadaires ($) |
| `Holiday_Flag` | Catégoriel | Indicateur de jour férié (0 = non, 1 = oui) |
| `Temperature` | Numérique | Température de la région (°F) |
| `Fuel_Price` | Numérique | Prix du carburant dans la région ($) |
| `CPI` | Numérique | Indice des prix à la consommation |
| `Unemployment` | Numérique | Taux de chômage de la région (%) |

Le dataset contient environ **150 observations** avec des **valeurs manquantes** dans
plusieurs colonnes, ce qui nécessite un traitement adapté.

## Méthodologie

### 1. Analyse exploratoire (EDA)

- Statistiques descriptives et identification des valeurs manquantes.
- Visualisations : distribution de la cible, matrice de corrélation, boxplots par magasin,
  scatterplots des features vs la cible.
- **Observation clé** : les ventes varient fortement d'un magasin à l'autre et sont plus
  élevées lors des jours fériés.

### 2. Préparation des données (Preprocessing)

- **Suppression des NaN sur la cible** : pas d'imputation sur `Weekly_Sales` pour éviter
  les biais.
- **Feature engineering temporel** : extraction de `Year`, `Month`, `Day`, `DayOfWeek`
  à partir de la colonne `Date`.
- **Suppression des outliers** : règle des 3 écarts-types (3σ) appliquée sur
  `Temperature`, `Fuel_Price`, `CPI`, `Unemployment`.
- **Pipeline scikit-learn** intégrant le preprocessing et le modèle dans un même objet
  `Pipeline` pour éviter le *data leak* lors de la validation croisée.

### 3. Modélisation

| Modèle | Description | Optimisation |
|--------|-------------|--------------|
| **Régression Linéaire** | Baseline — sans régularisation | — |
| **Ridge (L2)** | Pénalise les grands coefficients | GridSearchCV sur `alpha` |
| **Lasso (L1)** | Sélection automatique de features | GridSearchCV sur `alpha` |

**Choix de l'approche supervisée** : la variable cible est quantitative et des exemples
étiquetés sont disponibles → **régression supervisée**.

### 4. Évaluation

Trois métriques complémentaires ont été retenues :

| Métrique | Rôle |
|----------|------|
| **R²** | Proportion de la variance expliquée (pouvoir prédictif global) |
| **MAE** | Erreur moyenne en dollars (interprétabilité directe) |
| **RMSE** | Pénalise davantage les grandes erreurs (robustesse) |

La **validation croisée 5-fold** est utilisée pour estimer le pouvoir de généralisation
de chaque modèle.

## Résultats clés

- Les modèles **régularisés** (Ridge, Lasso) sont comparés à la régression linéaire ; sur ce
  **jeu de données limité**, les métriques test sont **très proches** entre baseline et
  modèles régularisés — **Ridge** peut légèrement devancer les autres sur le jeu de test.
  La régularisation illustre surtout une **méthode** de contrôle de la complexité et de
  recherche d'`alpha`, plus qu'un gain massif systématique.
- Le **GridSearchCV** a permis d'identifier une valeur pertinente de l'hyperparamètre `alpha`
  pour chaque modèle régularisé.
- Les **variables magasin** (`Store`) sont les plus influentes, suivies des indicateurs
  temporels et du `Holiday_Flag`.

## Recommandations business

1. **Adapter les budgets marketing par magasin** en fonction de leur potentiel de ventes.
2. **Intensifier les promotions avant les jours fériés** pour maximiser l'effet positif
   observé du `Holiday_Flag`.
3. **Tenir compte de la saisonnalité** (mois, jour de la semaine) dans la planification
   des stocks et des campagnes.
4. **Intégrer une veille économique** (CPI, chômage) pour anticiper les variations
   de consommation.

## Structure du repository

```
01-Walmart_sales/
├── 01-Walmart_sales.ipynb           # Énoncé
├── 01-Walmart_sales_solution.ipynb  # Corrigé complet (EDA → modélisation → résultats)
├── Walmart_Store_sales.csv          # Dataset
└── README.md                        # Ce fichier
```

## Reproduction

### Prérequis

- Python 3.8+
- Bibliothèques :

```
pandas
numpy
matplotlib
seaborn
scikit-learn
```

### Installation

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

### Exécution

1. Ouvrir **`01-Walmart_sales_solution.ipynb`** (corrigé) dans Jupyter Notebook / JupyterLab / VS Code. Pour ne travailler que sur l'énoncé, utiliser `01-Walmart_sales.ipynb`.
2. Exécuter toutes les cellules séquentiellement (`Kernel > Restart & Run All`).
3. Les résultats, graphiques et tableaux comparatifs s'affichent directement dans le notebook.

## Critères d'évaluation couverts (Grille Bloc 3)

| # | Critère | Comment c'est adressé |
|---|---------|----------------------|
| 1 | Pertinence du choix d'algorithme | Justification supervisé/régression dans l'introduction |
| 2 | Pipeline de préparation des données | `ColumnTransformer` + `Pipeline` scikit-learn |
| 3 | Qualité de l'optimisation | `GridSearchCV` avec 9 valeurs de `alpha` et CV 5-fold |
| 4 | Propreté du code | Structure lisible, sections markdown claires, code organisé (référence PEP8) |
| 5 | Performance de l'algorithme | Métriques R², MAE, RMSE sur train et test |
| 6 | Process d'analyses prédictives efficaces | Workflow structuré : EDA → Preprocessing → Modèle → Évaluation |
| 7 | Pertinence des critères d'évaluation | Choix justifié de R² + MAE + RMSE |
| 8 | Indicateurs meilleurs que version précédente | Comparaison baseline vs Ridge vs Lasso |
| 9 | Performances optimisées | GridSearchCV, visualisation R² vs alpha |
| 10 | Recommandations pertinentes | Analyse des coefficients + 4 recommandations business |

## Limites et pistes d'amélioration

- **Taille du dataset** limitée (~150 lignes) → résultats à consolider avec plus de données.
- **Modèle linéaire** : ne capture pas les relations non-linéaires.
- **Pistes** : Random Forest, XGBoost, features d'interaction, encodage cyclique des variables
  temporelles.
