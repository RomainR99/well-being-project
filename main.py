import pandas


def load_data(csv_path, target_col="target"):
		df = pandas.read_csv(csv_path)
		X = df.drop(columns=[target_col])
		Y = df[target_col]
		return X, Y 


if __name__ == "__main__":
	X, Y = load_data(csv_path="bienetre.csv")
	print(X.head())
	print(Y)