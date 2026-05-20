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