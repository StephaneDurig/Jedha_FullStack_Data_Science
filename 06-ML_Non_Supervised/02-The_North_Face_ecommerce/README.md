# The North Face E-commerce - Boosting Online Sales

Projet de Machine Learning Non-Supervise | Bloc 3 - JEDHA Data Science Fullstack

---

## Contexte et problematique

**The North Face** souhaite exploiter le Machine Learning pour ameliorer les ventes sur son site e-commerce. Deux solutions ont ete identifiees :

- **Systeme de recommandation** : suggerer des produits similaires ("vous pourriez aussi aimer...") pour augmenter le taux de conversion et le panier moyen
- **Extraction de topics** : ameliorer la structure du catalogue en identifiant des categories de produits pertinentes via l'analyse automatique des descriptions

**Objectifs** :
1. Identifier des groupes de produits avec des descriptions similaires (clustering)
2. Construire un systeme de recommandation simple base sur ces groupes
3. Extraire les topics latents des descriptions (topic modeling)

## Donnees

- **Source** : [Product Item Data (Kaggle)](https://www.kaggle.com/cclark/product-item-data?select=sample-data.csv)
- **Fichier** : `product-item-data/sample-data.csv`
- **Colonnes** : `id` (identifiant produit), `description` (description HTML du produit)
- **Volume** : ~500 produits du catalogue The North Face

## Methodologie

### 1. Pipeline de preprocessing NLP

- Nettoyage des balises HTML (`<br>`, `<b>`, `<ul>`, `<li>`, etc.) avec `re.sub()`
- Suppression de la ponctuation et passage en minuscules
- Tokenization et lemmatisation avec **spaCy** (`en_core_web_sm`)
- Suppression des stop words (`spacy.lang.en.stop_words.STOP_WORDS`)
- Filtrage des tokens courts (< 3 caracteres)

### 2. Vectorisation TF-IDF

- Transformation des textes preprocesses en vecteurs numeriques via `TfidfVectorizer`
- Matrice creuse documents x termes (max 5000 features)

### 3. Clustering DBSCAN (Partie 1)

- **V1** : DBSCAN avec parametres de base (eps=0.5, min_samples=5, distance cosine)
- **Optimisation** : methode kNN pour estimer epsilon + grille sur (eps, min_samples)
- **V2** : selection sous **contraintes alignees sur l'enonce** (fourchette 10-20 clusters, plafond sur la part d'outliers), puis **meilleure silhouette** (tie-break : moins d'outliers)
- **Analyse** : wordclouds par cluster pour interpreter les groupes de produits

### 4. Systeme de recommandation (Partie 2)

- Fonction `find_similar_items(item_id)` basee sur les clusters DBSCAN
- Classement des recommandations par similarite cosine au sein du meme cluster
- Gestion des outliers (recommandation par similarite globale)
- Interface interactive avec `input()`

### 5. Topic Modeling LSA (Partie 3)

- **V1** : TruncatedSVD avec 10 composantes
- **Optimisation** : analyse de la variance expliquee cumulee pour choisir le nombre optimal de topics
- **V2** : TruncatedSVD avec nombre de topics optimise
- Extraction du topic principal par document
- Wordclouds par topic pour interpreter les themes latents

## Choix techniques et justifications

| Choix | Justification |
|-------|---------------|
| **Apprentissage non-supervise** | Pas de variable cible, objectif de decouverte de structure dans les donnees textuelles |
| **DBSCAN** | Clusters de formes arbitraires, detection des outliers, pas besoin de fixer K a priori |
| **Distance cosine** | Metrique standard pour la comparaison de textes (insensible a la norme des vecteurs) |
| **LSA (TruncatedSVD)** | Extraction de topics sur matrice creuse, reduction de dimensionnalite semantique |
| **spaCy** | Lemmatisation et tokenization de qualite industrielle |
| **TF-IDF** | Penalise les termes trop communs, met en valeur les termes discriminants |
| **Silhouette Score** | Mesure objective de la qualite du clustering (cohesion intra-cluster, separation inter-cluster) |
| **Variance expliquee** | Critere d'evaluation standard pour le choix du nombre de composantes en SVD |
| **Precomputation distances** | Matrice de distances cosine precomputee pour optimiser le Grid Search |

## Structure du projet

```
02-The_North_Face_ecommerce/
├── 02-The_North_Face_ecommerce.ipynb           # Enonce du projet
├── 02-The_North_Face_ecommerce_solutions.ipynb  # Solution complete
├── README.md                                    # Ce fichier
└── product-item-data/
    └── sample-data.csv                          # Donnees produits
```

## Technologies utilisees

- **Python 3.10+**
- **pandas** / **numpy** : manipulation de donnees et calcul numerique
- **spaCy** (`en_core_web_sm`) : preprocessing NLP (tokenization, lemmatisation)
- **scikit-learn** : `TfidfVectorizer`, `DBSCAN`, `TruncatedSVD`, `silhouette_score`, `NearestNeighbors`, `pairwise_distances`, `cosine_similarity`
- **matplotlib** / **WordCloud** : wordclouds et visualisations statiques
- **plotly** : histogrammes et bar charts interactifs

## Execution

1. Ouvrir le notebook `02-The_North_Face_ecommerce_solutions.ipynb` dans Jupyter
2. Executer toutes les cellules sequentiellement

**Prerequis** :
```bash
pip install pandas numpy scikit-learn spacy plotly matplotlib wordcloud
python -m spacy download en_core_web_sm
```

## Resultats cles

- **Clustering DBSCAN** : identification de groupes de produits homogenes (vetements techniques, accessoires, sous-vetements...) avec V2 choisie dans une grille sous contraintes (clusters / outliers) puis silhouette
- **Systeme de recommandation** : la fonction `find_similar_items` produit des suggestions pertinentes basees sur les descriptions, y compris pour les produits outliers
- **Topic Modeling LSA** : extraction de topics interpretables correspondant a des categories de produits (outdoor, baselayers, bags, kids, etc.)
- **Amelioration iterative** : les modeles V2 s'appuient sur des criteres explicites (contraintes DBSCAN, variance expliquee pour LSA) pour affiner les hyperparametres par rapport aux baselines V1
