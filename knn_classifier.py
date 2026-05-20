class KNeighborsClassifier(KNeighborsMixin, ClassifierMixin, NeighborsBase):

    _parameter_constraints: dict = {**NeighborsBase._parameter_constraints}
    _parameter_constraints.pop("radius")
    _parameter_constraints.update(
        {"weights": [StrOptions({"uniform", "distance"}), callable, None]}
    )

    def __init__(
        self,
        n_neighbors=5,
        *,
        weights="uniform",
        algorithm="auto",
        leaf_size=30,
        p=2,
        metric="minkowski",
        metric_params=None,
        n_jobs=None,
    ):
        super().__init__(
            n_neighbors=n_neighbors,
            algorithm=algorithm,
            leaf_size=leaf_size,
            metric=metric,
            p=p,
            metric_params=metric_params,
            n_jobs=n_jobs,
        )
        self.weights = weights

    @_fit_context(
        prefer_skip_nested_validation=False
    )
    def fit(self, X, y):
        return self._fit(X, y)

    def predict(self, X):
        check_is_fitted(self, "_fit_method")
        if self.weights == "uniform":
            if self._fit_method == "brute" and ArgKminClassMode.is_usable_for(
                X, self._fit_X, self.metric
            ):
                probabilities = self.predict_proba(X)
                if self.outputs_2d_:
                    return np.stack(
                        [
                            self.classes_[idx][np.argmax(probas, axis=1)]
                            for idx, probas in enumerate(probabilities)
                        ],
                        axis=1,
                    )
                return self.classes_[np.argmax(probabilities, axis=1)]
            neigh_ind = self.kneighbors(X, return_distance=False)
            neigh_dist = None
        else:
            neigh_dist, neigh_ind = self.kneighbors(X)

        classes_ = self.classes_
        _y = self._y
        if not self.outputs_2d_:
            _y = self._y.reshape((-1, 1))
            classes_ = [self.classes_]

        n_outputs = len(classes_)
        n_queries = _num_samples(self._fit_X if X is None else X)
        weights = _get_weights(neigh_dist, self.weights)
        if weights is not None and _all_with_any_reduction_axis_1(weights, value=0):
            raise ValueError(
                "All neighbors of some sample is getting zero weights. "
                "Please modify 'weights' to avoid this case if you are "
                "using a user-defined function."
            )

        y_pred = np.empty((n_queries, n_outputs), dtype=classes_[0].dtype)
        for k, classes_k in enumerate(classes_):
            if weights is None:
                mode, _ = _mode(_y[neigh_ind, k], axis=1)
            else:
                mode, _ = weighted_mode(_y[neigh_ind, k], weights, axis=1)

            mode = np.asarray(mode.ravel(), dtype=np.intp)
            y_pred[:, k] = classes_k.take(mode)

        if not self.outputs_2d_:
            y_pred = y_pred.ravel()

        return y_pred

    def predict_proba(self, X):
        check_is_fitted(self, "_fit_method")
        if self.weights == "uniform":
            metric, metric_kwargs = _adjusted_metric(
                metric=self.metric, metric_kwargs=self.metric_params, p=self.p
            )
            if (
                self._fit_method == "brute"
                and ArgKminClassMode.is_usable_for(X, self._fit_X, metric)
                and not self.outputs_2d_
            ):
                if self.metric == "precomputed":
                    X = _check_precomputed(X)
                else:
                    X = validate_data(
                        self, X, accept_sparse="csr", reset=False, order="C"
                    )

                probabilities = ArgKminClassMode.compute(
                    X,
                    self._fit_X,
                    k=self.n_neighbors,
                    weights=self.weights,
                    Y_labels=self._y,
                    unique_Y_labels=self.classes_,
                    metric=metric,
                    metric_kwargs=metric_kwargs,
                    strategy="parallel_on_X",
                )
                return probabilities

            neigh_ind = self.kneighbors(X, return_distance=False)
            neigh_dist = None
        else:
            neigh_dist, neigh_ind = self.kneighbors(X)

        classes_ = self.classes_
        _y = self._y
        if not self.outputs_2d_:
            _y = self._y.reshape((-1, 1))
            classes_ = [self.classes_]

        n_queries = _num_samples(self._fit_X if X is None else X)

        weights = _get_weights(neigh_dist, self.weights)
        if weights is None:
            weights = np.ones_like(neigh_ind)
        elif _all_with_any_reduction_axis_1(weights, value=0):
            raise ValueError(
                "All neighbors of some sample is getting zero weights. "
                "Please modify 'weights' to avoid this case if you are "
                "using a user-defined function."
            )

        all_rows = np.arange(n_queries)
        probabilities = []
        for k, classes_k in enumerate(classes_):
            pred_labels = _y[:, k][neigh_ind]
            proba_k = np.zeros((n_queries, classes_k.size))

            for i, idx in enumerate(pred_labels.T):
                proba_k[all_rows, idx] += weights[:, i]

            normalizer = proba_k.sum(axis=1)[:, np.newaxis]
            proba_k /= normalizer

            probabilities.append(proba_k)

        if not self.outputs_2d_:
            probabilities = probabilities[0]

        return probabilities

    def score(self, X, y, sample_weight=None):
        return super().score(X, y, sample_weight)

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.classifier_tags.multi_label = True
        tags.input_tags.pairwise = self.metric == "precomputed"
        return tags
