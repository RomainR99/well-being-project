import random


class StratifiedKFold():

    def __init__(self, n_splits=5, shuffle=False, random_state=None):
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state

    def split(self, X, y):

        folds = [[] for _ in range(self.n_splits)]

        for current_label in set(y):

            current_indexes = []

            for position, current_value in enumerate(y):

                if current_value == current_label:
                    current_indexes.append(position)

            if self.shuffle:
                random.shuffle(current_indexes)

            division_size = len(current_indexes) // self.n_splits

            for fold_number in range(self.n_splits):

                start = fold_number * division_size

                if fold_number == self.n_splits - 1:
                    end = len(current_indexes)
                else:
                    end = start + division_size

                folds[fold_number].extend(
                    current_indexes[start:end]
                )

        return folds
