import pandas
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score

def load_data(csv_path, target_col="target"):
	df = pandas.read_csv(csv_path)
	print(df[target_col].value_counts())
	X = df.drop(columns=[target_col])
	Y = df[target_col]
	return X, Y 

def nomalize(X):
	standard_scaler_object = StandardScaler()
	X_normalized = standard_scaler_object.fit_transform(X)
	return standard_scaler_object, X_normalized


def evaluate(X_normalized, Y, n_neighbors=5, n_splits=10, scoring="f1_weighted"):
	cross_validation_object = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
	knn_object = KNeighborsClassifier(n_neighbors=n_neighbors)

	scores = cross_val_score(knn_object, X_normalized, Y, cv=cross_validation_object, scoring=scoring)
	print(f"N neighbors : {n_neighbors}")
	print(f"{scoring} moyen {scores.mean()} +/- {scores.std()}")
	print("-"*20)
	return scores


def find_optimal_kvalue(X_normalized, Y, n_splits=10, scoring="f1_weighted", n_neighbors_range=range(1, 100,2)):
	"""Grid Search CV"""

	dict_of_k_scores = {}
	
	for n_neighbors in n_neighbors_range:
		scores = evaluate(X_normalized, Y, n_neighbors=n_neighbors, n_splits=n_splits, scoring=scoring)
		current_mean_score = scores.mean()

		dict_of_k_scores[n_neighbors] = current_mean_score

	optimal_n_neighbors = max(dict_of_k_scores, key=dict_of_k_scores.get)
	print("-"*100)
	print(f"The optimal N neighbors value is {optimal_n_neighbors}")
	print("-"*100)
	return optimal_n_neighbors


if __name__ == "__main__":
	X, Y = load_data(csv_path="bienetre.csv")
	standard_scaler_object, X_normalized = nomalize(X)

	optimal_n_neighbors = find_optimal_kvalue(X_normalized, Y, scoring="f1_weighted")