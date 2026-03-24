# Conversion Rate Challenge — Prédiction des conversions

## Description du projet

Le site **datascienceweekly.org** est une newsletter populaire animée par des data scientists
indépendants. L'équipe éditoriale souhaite comprendre le comportement des visiteurs de leur
site et déterminer s'il est possible de **prédire si un visiteur va s'abonner à la newsletter**
(= *conversion*) à partir de quelques informations le concernant.

Ce projet construit un modèle de **classification binaire supervisée** optimisé sur le
**F1-score**, métrique adaptée au fort déséquilibre des classes (~3 % de conversions).

## Dataset

| Fichier | Description |
|---------|-------------|
| `conversion_data_train.csv` | Données étiquetées (~284 000 lignes) |
| `conversion_data_test.csv` | Données non étiquetées (~31 600 lignes) |

**Variables :**

| Colonne | Type | Description |
|---------|------|-------------|
| `country` | Catégoriel | Pays du visiteur (US, UK, China, Germany) |
| `age` | Numérique | Âge du visiteur |
| `new_user` | Binaire | Nouveau visiteur (1) ou récurrent (0) |
| `source` | Catégoriel | Source d'acquisition (Seo, Ads, Direct) |
| `total_pages_visited` | Numérique | Nombre de pages consultées |
| `converted` | Binaire | **Variable cible** — Abonnement (1) ou non (0) |

## Méthodologie

### 1. Analyse exploratoire (EDA)

- Distribution de la cible : fort déséquilibre (~97 % négatifs, ~3 % positifs).
- Visualisations : taux de conversion par pays, par source, par type d'utilisateur,
  distribution de l'âge par classe, boxplots de `total_pages_visited`.
- Matrice de corrélation des variables numériques.
- **Observation clé** : `total_pages_visited` est le facteur le plus discriminant.

### 2. Préparation des données (Preprocessing)

- **Split stratifié** 80/20 pour préserver la proportion des classes.
- **Pipeline scikit-learn** avec `ColumnTransformer` :
  - Numériques (`age`, `total_pages_visited`) : `SimpleImputer(median)` → `StandardScaler`
  - Catégorielles (`country`, `source`) : `SimpleImputer(most_frequent)` → `OneHotEncoder(drop='first')`
  - Binaire (`new_user`) : passthrough

### 3. Modélisation

| Modèle | Description | Optimisation |
|--------|-------------|--------------|
| **Régression logistique** | Baseline — sans tuning | — |
| **Régression logistique régularisée** | Pénalisation L2 | GridSearchCV sur `C` |
| **Arbre de décision** | Modèle non-linéaire | GridSearchCV sur `max_depth`, `min_samples_split` |
| **Random Forest** | Ensemble d'arbres (bagging) | GridSearchCV sur `n_estimators`, `max_depth` |
| **Gradient Boosting** | Ensemble d'arbres (boosting) | GridSearchCV sur `n_estimators`, `learning_rate`, `max_depth` |

**Choix de l'approche supervisée** : la variable cible est binaire et des exemples étiquetés
sont disponibles → **classification binaire supervisée**.

### 4. Évaluation

| Métrique | Rôle |
|----------|------|
| **F1-score** | Métrique principale — moyenne harmonique précision/rappel |
| **Précision** | Proportion de vrais positifs parmi les prédictions positives |
| **Rappel** | Proportion de vrais positifs détectés |
| **Matrice de confusion** | Visualisation des erreurs de classification |

La **validation croisée 5-fold** est utilisée pour estimer la capacité de généralisation.

## Résultats clés

- Le modèle **baseline** (régression logistique) sert de point de référence.
- Les modèles **tree-based** (Random Forest, Gradient Boosting) sont comparés sur le F1-test : selon le tirage train/test, la baseline peut rester la meilleure — le modèle retenu est **toujours** celui qui maximise le F1 sur le jeu de test parmi les candidats.
- Le **GridSearchCV** a permis d'optimiser les hyperparamètres de chaque modèle.
- La feature `total_pages_visited` est de loin la plus importante, suivie de `new_user`
  et `age`.

## Recommandations business

1. **Optimiser l'expérience de navigation** : encourager la consultation de multiples pages
   (articles recommandés, liens internes, contenu engageant).
2. **Cibler les anciens utilisateurs** : investir dans des campagnes de réengagement
   (emails de relance, notifications) pour les visiteurs récurrents qui convertissent mieux.
3. **Adapter la stratégie par pays** : localiser le contenu ou proposer des offres ciblées
   par zone géographique.
4. **Segmenter par tranche d'âge** : personnaliser le message marketing selon l'âge du visiteur.
5. **Diversifier les sources d'acquisition** : réallouer le budget vers les canaux les plus
   performants en termes de conversion.

## Structure du repository

```
02-Conversion_rate_challenge/
├── 02-Conversion_rate_challenge.ipynb              # Notebook principal
├── conversion_data_train.csv                       # Dataset d'entraînement
├── conversion_data_test.csv                        # Dataset de test (sans labels)
├── conversion_data_test_predictions_challenge.csv  # Prédictions exportées
└── README.md                                       # Ce fichier
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
plotly
```

### Installation

```bash
pip install pandas numpy matplotlib seaborn scikit-learn plotly
```

### Exécution

1. Ouvrir `02-Conversion_rate_challenge.ipynb` dans Jupyter Notebook / JupyterLab / VS Code.
2. Exécuter toutes les cellules séquentiellement (`Kernel > Restart & Run All`).
3. Les résultats, graphiques et tableaux comparatifs s'affichent dans le notebook.
4. Le fichier de prédictions est généré automatiquement.

## Limites et pistes d'amélioration

- **Déséquilibre des classes** : les modèles scikit-learn utilisés permettent d'explorer le paramètre `class_weight` (pondération des classes) pour mieux prendre en compte la classe minoritaire, dans le même pipeline.
- **Feature engineering** : enrichir les entrées (interactions simples, tranches d'âge, regroupements de catégories) pourrait améliorer le signal sans changer l'architecture générale du preprocessing.
- **Hyperparamètres** : élargir ou affiner les valeurs testées dans `GridSearchCV` sur les familles de modèles déjà choisies.
