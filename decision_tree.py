from data_loader import load_normalized_data
from tqdm import tqdm


class DecisionTree:

	def __init__(self, max_depth=6, min_samples_leaf=20, #max_depth=6 on peut mettre 8 apres risque d'overfitting
                                                        #20 personnes minimum par feuille, leaf = feuille
                 min_information_gain=0.0, #combien le split rend les groupes plus "purs"
                 numb_of_features_splitting=None,#L’arbre regarde toutes les colonnes.
                                            #Random Forest on pourra limiter les features
                 amount_of_say=None) -> None: #le poids d’un arbre dans le vote final boosting :plusieurs petits arbres travaillent ensemble pour créer un modèle plus puissant.
                 #-> None “cette fonction ne retourne rien”.
		self.max_depth = max_depth
		self.min_samples_leaf = min_samples_leaf
		self.min_information_gain = min_information_gain
		self.numb_of_features_splitting = numb_of_features_splitting
		self.amount_of_say = amount_of_say


	# Phase 2 — Prédiction.
	# Pour un nouveau point, on cherche ses K voisins les plus proches dans les
	# données d'entraînement, puis on fait un vote pondéré.
	def predict(self, x_to_forcast):
		# Garde-fou : si fit() n'a pas été appelé, les attributs n'existent pas.
		# hasattr() vérifie l'existence d'un attribut sans lever d'erreur.
		if not hasattr(self, "X_normalized") or not hasattr(self, "Y"):
			raise Exception("The model is not fit")

		# Étape 1 — Calculer la distance entre x_to_forcast et CHAQUE point connu.
		# enumerate() fournit à la fois l'index et la valeur à chaque itération.
		# tqdm affiche une barre de progression (utile : cette boucle peut être longue).
		distances = {}
		for index_x, current_x in tqdm(enumerate(self.X_normalized), total=len(self.X_normalized)):
			current_distance = KNN.euclidian_distance(current_x, x_to_forcast)
			# Arrondi à 2 décimales : évite que 1.000001 et 1.000002 soient traités
			# comme des distances différentes alors qu'elles sont pratiquement égales.
			distances[index_x] = float(f"{current_distance:.2f}")

		# Étape 2 — Trier les points du plus proche au plus loin.
		sorted_distances = KNN.sorted_dict_by_values(distances)

		# Étape 3 — Garder uniquement les K premiers index (les K plus proches voisins).
		# [:self.n_neighbors] est un slice : prend les n premiers éléments de la liste.
		indexes_nearest_neighbors = list(sorted_distances.keys())[: self.n_neighbors]

		# Étape 4 — Vote pondéré : chaque voisin vote pour sa classe.
		# Le poids compense le déséquilibre : une classe rare vote plus fort.
		label_counter = {label : 0 for label in self.set_of_labels}
		for index_point in indexes_nearest_neighbors:
			label = self.Y[index_point]
			label_counter[label] += self.label_weight[label]

		# Étape 5 — La classe avec le score total le plus élevé est la prédiction.
		# Le dictionnaire est trié par ordre croissant, donc le gagnant est le dernier [-1].
		sorted_label_counter = KNN.sorted_dict_by_values(label_counter)
		predicted_label = list(sorted_label_counter.keys())[-1]

		return predicted_label


	# @staticmethod : cette méthode n'utilise pas self, elle ne dépend d'aucun
	# attribut de l'objet. C'est un outil utilitaire lié conceptuellement à KNN.
	# On peut l'appeler directement : KNN.euclidian_distance(x1, x2)
	@staticmethod
	def euclidian_distance(x1, x2):
		x1, x2 = list(x1), list(x2)[0]  # conversion en listes Python standard

		# Formule : √[ Σ(aᵢ - bᵢ)² ]
		# zip(strict=True) associe les features deux à deux et lève une erreur
		# si les deux points n'ont pas le même nombre de dimensions.
		return sum([(a-b)**2 for a, b in zip(x1, x2, strict=True)])**(1/2)


	# Trie un dictionnaire par ses valeurs (ordre croissant).
	# lambda est une fonction anonyme : lambda item: item[1] extrait la valeur
	# de chaque paire (clé, valeur) pour l'utiliser comme critère de tri.
	@staticmethod
	def sorted_dict_by_values(dict_object):
		sorted_dict = dict(sorted(dict_object.items(), key=lambda item: item[1]))
		return sorted_dict





if __name__ == "__main__":
	X_normalized, Y, standard_scaler_object = load_normalized_data(file_path="bienetre.csv")
	knn_object = KNN(n_neighbors=7)
	knn_object.fit(X_normalized, Y)
	print(knn_object.predict([X_normalized[7]]))
	print(Y[7])