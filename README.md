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

## KNN — ce que fait `fit()`

Pour KNN, contrairement à d’autres modèles, `fit()` **n’apprend pas vraiment une formule**. Il stocke simplement les exemples connus (lazy learning).

```python
knn_object.fit(X_normalized, Y)
```

veut dire :

- **Voici les personnes connues :** `X_normalized`
- **Voici leurs classes :** `Y`
- **Garde-les en mémoire** pour comparer plus tard lors de `predict()`.

Tout le « apprentissage » se fait au moment de la prédiction : on calcule les distances vers chaque exemple mémorisé, on garde les `k` plus proches voisins, puis on vote.

Implémentation : `knn_hakim.py`.

## Arbre de décision (Decision Tree)

### `fit()` — pas comme le KNN

Contrairement au KNN, l’arbre de décision **ne calcule pas de distances** entre points. Il compare des seuils sur chaque feature (`stress <= 6`, etc.).

Donc **pas besoin de données normalisées** pour `fit()` :

```python
def fit(self, X, Y):
    ...
```

| Modèle | `fit()` attend souvent | Pourquoi |
|--------|------------------------|----------|
| **KNN** | `X_normalized` | distances euclidiennes sensibles à l’échelle des colonnes |
| **Decision Tree** | `X` (brut ou normalisé) | splits sur `<=` / `>` : multiplier une colonne par une constante ne change pas l’ordre des splits |

Dans `decision_tree.py`, on peut donc passer `X` directement après `load_data()`, sans `StandardScaler`.

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

**Important :** c’est **juste une annotation du code**. Python **ne l’applique pas** à l’exécution : tu peux écrire `-> str` et retourner un nombre — le programme tournera quand même. Les annotations servent aux **humains**, à l’**IDE** et aux **linters** (basedpyright, mypy) pour comprendre et vérifier le code.

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

### Utilité concrète n°3 : détecter des erreurs

Les annotations servent aussi à **détecter des erreurs** avant l’exécution (via l’IDE ou un linter comme basedpyright).

**Exemple :**

```python
def get_name() -> str:
    return "Romain"

name = get_name()

print(name + 5)  # erreur : on ne peut pas additionner str + int
```

L’IDE peut alors t’alerter :

> attention : `str` + `int`

Sans annotation, `get_name()` pourrait être confondu avec une fonction qui retourne autre chose ; avec `-> str`, l’outil sait que `name` est une chaîne et signale l’opération invalide.

En résumé : annotation = **documentation + aide à la détection d’erreurs**, pas une règle imposée par Python au runtime.

### Ce que tu dois observer

Quand tu coderas, compare les scores **train** et **test** :

| Train | Test | Interprétation |
|-------|------|----------------|
| 99 % | 55 % | → arbre **trop complexe** (sur-apprentissage) |
| 80 % | 78 % | → souvent **meilleur** (bonne généralisation) |

Un grand écart train/test signale que le modèle a appris le jeu d’entraînement par cœur plutôt que des règles utiles sur de nouvelles données.

## Notes Python — `keys()`, `values()`, `items()`

`keys()`, `values()` et `items()` sont **natifs Python**, mais **uniquement pour les dictionnaires** (`dict`), pas pour les `set`.

### 1. Dictionnaire (`dict`)

```python
personne = {
    "nom": "Romain",
    "age": 25
}
```

**`keys()`** — donne les clés :

```
"nom", "age"
```

**`values()`** — donne les valeurs :

```
"Romain", 25
```

**`items()`** — donne **clé + valeur** (paires) :

```python
("nom", "Romain")
("age", 25)
```

Utile pour parcourir un dict : `for cle, valeur in personne.items():`

### 2. Set (`set`)

Un `set` est différent :

```python
labels = {0, 1, 2}
```

Un set contient seulement des **valeurs uniques**. Il n’y a **pas** :

- de clé ;
- de valeur associée à une clé.

Donc :

```python
labels.keys()  # ❌ AttributeError
```

**Pourquoi ?** Parce qu’un `set` n’est pas une structure **clé → valeur**, contrairement à un `dict`.

Dans le projet (ex. `knn_hakim.py`), `set(Y)` sert à lister les **classes distinctes** ; les comptages par classe passent par un **dictionnaire** (`label_occurences`), où `keys()`, `values()` et `items()` ont du sens.

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