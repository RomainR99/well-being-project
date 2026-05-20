class KNeighborsClassifier:

    def __init__(self, n_neighbors=5):

        self.n_neighbors = n_neighbors
        self.training_data = None #initialises à None.
        self.training_labels = None

    def fit(self, X, y):
        self.training_data = X
        self.training_labels = y
        return self

    def _compute_distance(self, point_a, point_b):
        squared_sum = 0
        for a, b in zip(point_a, point_b):
            squared_sum += (a - b) ** 2
        return squared_sum ** 0.5

    def predict(self, X):

        # check_is_fitted(self, "_fit_method")
        # Supprimé : sklearn vérifie que fit() a été appelé.
        # Nous, on vérifie simplement training_data et training_labels.
        if self.training_data is None or self.training_labels is None:
            raise ValueError("The model must be trained with fit() before predict().")

        predictions = []

        for unknown_point in X:

            # if self.weights == "uniform":
            # Supprimé : on ne gère pas weights="distance".
            # Notre version utilise toujours un vote simple.

            # if self._fit_method == "brute" and ArgKminClassMode.is_usable_for(...)
            # Supprimé : optimisation sklearn/Cython.
            # Notre version fait toujours une recherche brute avec une boucle.

            # probabilities = self.predict_proba(X)
            # Supprimé : on ne calcule pas les probabilités.
            # On veut seulement la classe finale.

            # return self.classes_[np.argmax(probabilities, axis=1)]
            # Supprimé : nécessite numpy et les probabilités.
            # Nous faisons un vote majoritaire.

            # neigh_ind = self.kneighbors(X, return_distance=False)
            # Remplacé par : calcul manuel des distances.
            distances = []

            for position, known_point in enumerate(self.training_data):

                distance = self._compute_distance(unknown_point, known_point)
                label = self.training_labels[position]

                distances.append((distance, label))

            # neigh_dist = None
            # Supprimé : on ne garde pas une variable séparée pour les distances.
            # Elles sont déjà dans la liste distances.

            # classes_ = self.classes_
            # Supprimé : on utilise directement les labels stockés dans training_labels.

            # _y = self._y
            # Supprimé : notre équivalent est self.training_labels.

            # if not self.outputs_2d_:
            # Supprimé : on ne gère pas les sorties multi-dimensions.

            # n_outputs = len(classes_)
            # Supprimé : une seule sortie à prédire.

            # n_queries = _num_samples(...)
            # Supprimé : on boucle directement sur X.

            # weights = _get_weights(neigh_dist, self.weights)
            # Supprimé : pas de pondération par distance.

            # if weights is not None ...
            # Supprimé : pas de gestion des poids invalides.

            # y_pred = np.empty(...)
            # Supprimé : pas de tableau numpy.
            # On utilise une liste Python : predictions.

            distances.sort(key=lambda pair: pair[0])

            nearest_neighbors = distances[:self.n_neighbors]

            # mode, _ = _mode(...)
            # Remplacé par : dictionnaire de votes.
            votes = {}

            for distance, label in nearest_neighbors:

                if label not in votes:
                    votes[label] = 0

                votes[label] += 1

            # weighted_mode(...)
            # Supprimé : pas de vote pondéré.

            # mode = np.asarray(...)
            # Supprimé : pas de numpy.

            # y_pred[:, k] = classes_k.take(mode)
            # Remplacé par : classe avec le plus de votes.
            predicted_label = max(votes, key=votes.get)

            predictions.append(predicted_label)

        # if not self.outputs_2d_:
        #     y_pred = y_pred.ravel()
        # Supprimé : pas de sortie 2D.

        # return y_pred
        # Remplacé par :
        return predictions

    def predict_proba(self, X):

        # check_is_fitted(self, "_fit_method")
        # Supprimé : vérification avancée sklearn.
        # Nous vérifions simplement fit().
        if self.training_data is None or self.training_labels is None:
            raise ValueError(
                "Le modèle doit être entraîné avec fit() avant predict_proba()."
            )

        probabilities = []

        for unknown_point in X:

            # if self.weights == "uniform":
            # Supprimé : nous utilisons uniquement des votes simples.

            # metric, metric_kwargs = _adjusted_metric(...)
            # Supprimé : nous utilisons uniquement
            # la distance euclidienne classique.

            # if self._fit_method == "brute"
            # Supprimé : sklearn choisit plusieurs algorithmes.
            # Nous utilisons toujours une boucle simple.

            # ArgKminClassMode.is_usable_for(...)
            # Supprimé : optimisation interne sklearn/Cython.

            # if self.metric == "precomputed"
            # Supprimé : nous ne gérons pas les matrices
            # de distances pré-calculées.

            # validate_data(...)
            # Supprimé : validation avancée sklearn.

            # probabilities = ArgKminClassMode.compute(...)
            # Supprimé : calcul optimisé sklearn.

            # neigh_ind = self.kneighbors(...)
            # neigh_dist = None
            # Remplacé par : calcul manuel des distances.
            distances = []

            for position, known_point in enumerate(self.training_data):

                distance = self._compute_distance(
                    unknown_point,
                    known_point
                )

                label = self.training_labels[position]

                distances.append((distance, label))

            distances.sort(key=lambda pair: pair[0])

            nearest_neighbors = distances[:self.n_neighbors]

            # classes_ = self.classes_
            # _y = self._y
            # Supprimé : nous utilisons directement
            # self.training_labels.

            # outputs_2d_
            # Supprimé : pas de gestion multi-sorties.

            # n_queries = _num_samples(...)
            # Supprimé : nous parcourons directement X.

            # weights = _get_weights(...)
            # Supprimé : pas de poids par distance.

            # np.ones_like(...)
            # Supprimé : pas de numpy.

            # all_rows = np.arange(...)
            # Supprimé : pas de numpy.

            # pred_labels = _y[:, k][neigh_ind]
            # Supprimé : pas de slicing numpy.

            # proba_k = np.zeros(...)
            # Supprimé : pas de matrices numpy.

            votes = {}

            for distance, label in nearest_neighbors:

                if label not in votes:
                    votes[label] = 0

                votes[label] += 1

            total_votes = len(nearest_neighbors)

            current_probabilities = {}

            for label, count in votes.items():

                probability = count / total_votes

                current_probabilities[label] = probability

            probabilities.append(current_probabilities)

            # normalizer = proba_k.sum(...)
            # proba_k /= normalizer
            # Supprimé : notre division count / total_votes
            # normalise déjà les probabilités.

        # if not self.outputs_2d_
        # probabilities = probabilities[0]
        # Supprimé : pas de sortie 2D.

        return probabilities


# def score(self, X, y, sample_weight=None):
#     return super().score(X, y, sample_weight)
#
# Supprimé :
# sklearn calcule automatiquement accuracy.
# Nous pouvons créer une méthode evaluate() plus simple.


# def __sklearn_tags__(self):
#     tags = super().__sklearn_tags__()
#     tags.classifier_tags.multi_label = True
#     tags.input_tags.pairwise = self.metric == "precomputed"
#     return tags
#
# Supprimé :
# utilisé uniquement par l’écosystème interne sklearn.
# Inutile dans un projet from scratch.

    def score(self, X, y):

        # return super().score(X, y, sample_weight)
        # Supprimé :
        # sklearn calcule automatiquement le score
        # avec les classes parentes.
        # Nous n'avons pas de classe parent.

        predictions = self.predict(X)

        correct_predictions = 0

        for prediction, real_label in zip(predictions, y):

            if prediction == real_label:
                correct_predictions += 1

        accuracy = correct_predictions / len(y)

        return accuracy


# def __sklearn_tags__(self):
#
#     tags = super().__sklearn_tags__()
#
#     tags.classifier_tags.multi_label = True
#
#     tags.input_tags.pairwise = self.metric == "precomputed"
#
#     return tags
#
#
# Supprimé complètement :
#
# __sklearn_tags__ sert uniquement à communiquer
# des informations internes à sklearn :
#
# - multi-label
# - pairwise distance
# - sparse matrix
# - compatibilité pipeline
# - metadata sklearn
#
# Notre projet from scratch n'utilise aucun
# mécanisme interne sklearn.
#
# Donc cette méthode est inutile.
