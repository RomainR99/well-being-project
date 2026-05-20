class GridSearchCV:
    def __init__(
        self,
        model,
        param_grid,
        scoring=None,
        cv=None,
    ):



        self.model = model

        self.param_grid = param_grid

        self.scoring = scoring

        self.cv = cv

        self.best_score_ = None

        self.best_params_ = None

    def fit(self, X, y):

        best_score = -1

        best_params = None

        # evaluate_candidates(ParameterGrid(...))
        # Supprimé :
        # sklearn génère automatiquement toutes
        # les combinaisons de paramètres.
        #
        # Nous faisons simplement une boucle.

        for current_k in self.param_grid["n_neighbors"]:

            scores = []

            # cv.split(...)
            # Conservé :
            # cross validation manuelle.
            for train_indexes, test_indexes in self.cv.split(X, y):

                X_train = []
                y_train = []

                X_test = []
                y_test = []

                for index in train_indexes:
                    X_train.append(X[index])
                    y_train.append(y[index])

                for index in test_indexes:
                    X_test.append(X[index])
                    y_test.append(y[index])

                # model.set_params(...)
                # Supprimé :
                # nous modifions directement n_neighbors.
                self.model.n_neighbors = current_k

                self.model.fit(X_train, y_train)

                score = self.model.score(X_test, y_test)

                scores.append(score)

            average_score = sum(scores) / len(scores)

            if average_score > best_score:

                best_score = average_score

                best_params = {
                    "n_neighbors": current_k
                }

        self.best_score_ = best_score

        self.best_params_ = best_params

        return self


# class RandomizedSearchCV(BaseSearchCV):
#
# Supprimé complètement :
#
# RandomizedSearchCV teste des paramètres aléatoires.
#
# Ton exercice utilise seulement :
#
# GridSearchCV
#
# donc cette classe est inutile.
#
# param_distributions
# n_iter
# random_state
# ParameterSampler
#
# deviennent inutiles également.
