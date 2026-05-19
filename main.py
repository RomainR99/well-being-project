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


def evaluate(X_normalized, Y, n_neighbors=5, n_splits=10):
	cross_validation_object = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
	knn_object = KNeighborsClassifier(n_neighbors=n_neighbors)

	scores = cross_val_score(knn_object, X_normalized, Y, cv=cross_validation_object, scoring="f1_weighted")
	print(f"F1-score moyen {scores.mean()} +/- {scores.std()}")
	return scores


if __name__ == "__main__":
	X, Y = load_data(csv_path="bienetre.csv")
	standard_scaler_object, X_normalized = nomalize(X)
	scores = evaluate(X_normalized, Y, n_neighbors=5, n_splits=10)