# Well-Being Project

Projet d'introduction au machine learning appliqué à la classification du bien-être.

## Description

Ce projet met en œuvre un pipeline de classification basé sur l'algorithme **K-Nearest Neighbors (KNN)** sur un jeu de données de bien-être (`bienetre.csv`). Il inclut une recherche automatique de la valeur optimale de `k` via une grille de recherche artisanale avec validation croisée.

## Pipeline

1. **Chargement des données** (`load_data`) — lecture du CSV, affichage de la distribution des classes (équilibre du dataset), séparation features/cible.
2. **Normalisation** (`nomalize`) — standardisation des features avec `StandardScaler`.
3. **Évaluation** (`evaluate`) — validation croisée stratifiée (`StratifiedKFold`, 10 folds) avec scoring `f1_weighted`.
4. **Recherche de k optimal** (`find_optimal_kvalue`) — grid search artisanal sur `k ∈ [1, 99]` (valeurs impaires), sélection du meilleur `k` selon le F1 moyen.

## Utilisation

```bash
python main.py
```

Le script charge `bienetre.csv`, normalise les données, puis cherche la valeur de `k` optimale et affiche les résultats.

## Résultats

Exécution de `python3 main.py` (KNN from scratch, grille `[3, 5, 7, 9]`, `n_splits=5`) :

```
Loading data phase
target
1    4000
0    4000
2    2000
Name: count, dtype: int64
Sucess : Loading data
--------------------
Finding optimal params for KNeighborsClassifier
Best Params
{'n_neighbors': 5}
f1_weighted : 0.9986
----------------------------------------------------------------------------------------------------
Le model le plus optimisé est KNeighborsClassifier
Best Params : {'n_neighbors': 5}
f1_weighted : 0.9986
```

| Métrique | Valeur |
|----------|--------|
| Modèle | `KNeighborsClassifier` |
| Meilleur `n_neighbors` | **5** |
| Score `f1_weighted` | **0.9986** |

## Arbre de décision (Decision Tree)

Sur ce jeu de données de bien-être, on observe typiquement :

- des **variables très corrélées** ;
- du **bruit** ;
- des **relations complexes**.

Exemples de corrélations entre features :

| Paire | Lien |
|-------|------|
| poids ↔ IMC | mesures redondantes du même phénomène |
| stress ↔ sommeil | variables liées au rythme de vie |
| activité ↔ exercice | indicateurs proches de l’effort physique |

Un arbre **trop profond** risque alors :

- de **mémoriser des cas particuliers** (sur-apprentissage) ;
- au lieu d’**apprendre de vraies tendances** généralisables.

D’où l’intérêt de limiter la profondeur (`max_depth`), d’imposer un gain d’information minimal (`min_information_gain`) ou un minimum d’échantillons par feuille (`min_samples_leaf`) — paramètres exposés dans `decision_tree.py`.

### `min_information_gain`

Mesure **combien le split rend les groupes plus « purs »** : on n’accepte une division que si le gain d’information dépasse ce seuil. Plus la valeur est haute, moins l’arbre se divise.

**Valeurs réalistes** (souvent) :

| Valeur | Effet |
|--------|--------|
| `0.0` | accepte tous les splits |
| `0.001` | filtre très léger |
| `0.01` | filtre raisonnable |
| `0.1` | assez strict |
| `1` | souvent trop énorme |

### `-> None` (annotation de type)

`-> None` est une **annotation de type** en Python. Elle indique : *« cette fonction ne retourne rien »*.

**Exemple :**

```python
def bonjour() -> None:
    print("Salut")
```

Ici :

- la fonction **affiche** quelque chose ;
- mais **ne retourne aucune valeur** utile.

Donc `-> None` signifie : pas de `return` utile (équivalent implicite à `return None` si on ne précise rien).

**Cas du constructeur** (`__init__` dans `decision_tree.py`) :

```python
def __init__(self, max_depth=4, ...) -> None:
    self.max_depth = max_depth
    ...
```

Le constructeur :

- **initialise** l’objet ;
- **stocke** des variables (`self.max_depth`, etc.) ;
- mais **ne renvoie rien** — c’est normal pour `__init__`.

### Ce que tu dois observer

Quand tu coderas, compare les scores **train** et **test** :

| Train | Test | Interprétation |
|-------|------|----------------|
| 99 % | 55 % | → arbre **trop complexe** (sur-apprentissage) |
| 80 % | 78 % | → souvent **meilleur** (bonne généralisation) |

Un grand écart train/test signale que le modèle a appris le jeu d’entraînement par cœur plutôt que des règles utiles sur de nouvelles données.

## Dépendances

- Python 3.x
- pandas
- scikit-learn

## Ressources

- [Decision and Classification Trees, Clearly Explained!!!](https://www.youtube.com/watch?v=_L39rN6gz7Y)

attention au outliers avec k=1 et k=2 d'ou k=5 à 100 pas de 2.
Exemple :

class Chien:

    def __init__(self, nom):
        self.nom = nom

    def parler(self):
        print(self.nom)

Puis :

animal = Chien("Rex")
animal.parler()

Python fait en réalité :

Chien.parler(animal)