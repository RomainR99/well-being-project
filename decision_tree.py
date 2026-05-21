from data_loader import load_normalized_data
from tqdm import tqdm


class DecisionTree:

	def __init__(self, max_depth=6, min_samples_leaf=20, #max_depth=6 on peut mettre 8 apres risque d'overfitting
		#20 personnes minimum par feuille, leaf = feuille
		min_information_gain=0.0, #combien le split rend les groupes plus "purs"
		numb_of_features_splitting=None,#L’arbre regarde toutes les colonnes.
		#Random Forest on pourra limiter les features
		amount_of_say=None, criterion="gini") -> None: #le poids d’un arbre dans le vote final boosting :plusieurs petits arbres travaillent ensemble pour créer un modèle plus puissant.
		#-> None “cette fonction ne retourne rien”.
		self.max_depth = max_depth
		self.min_samples_leaf = min_samples_leaf
		self.min_information_gain = min_information_gain
		self.numb_of_features_splitting = numb_of_features_splitting
		self.amount_of_say = amount_of_say
		self.criterion = criterion #gini ou entropy

	#pas comme KNN , pas de calcul de distance donc pas besoin de normalisé
	def fit(self, X, Y):
		self.X = X
		self.Y = Y
		self.set_of_labels = set(Y)  #Sauvegarde les labels
		#le premier nœud de mon arbre est root
		self.root = self.build_tree(X, Y, depth=0)

	def build_tree(self, X, Y, depth):
		#nombre de personnes dans le groupe OU profondeur max atteinte
		#len(Y) = combien il y a de personnes dans cette liste
		if depth >= self.max_depth or len(Y) < self.min_samples_leaf:
			#"value" reçoit la classe finale et apres j'appelle : node.value
			#Node-question contient : feature, threshold, left, right, value
			#threshold le seuil de séparation
			#ex:stress <= 6 ? Élément	Valeur
			#feature   stress
			#threshold	6
			return Node(value=self.predict_leaf(Y))#predict_leafQuelle est la classe majoritaire dans Y
		#1. trouver le meilleur split
		best_split = self.find_best_split(X, Y)
		#2. séparer les personnes
		left_X, left_Y, right_X, right_Y = self.split_data(X, Y, best_split)
		#3 branche gauche
		left_node = self.build_tree(left_X, left_Y, depth + 1)
		right_node = self.build_tree(right_X, right_Y, depth + 1)
		#4. créer le node
		return Node(feature=best_split[0], threshold=best_split[1], left=left_node, right=right_node)

    def predict_leaf(self, Y):
            return Y.mode()[0]
            #mode() renvoie la valeur la plus fréquente dans Y
    
    
    def split_data(self, X, Y, best_split):
        left_X = X[X[best_split[0]] <= best_split[1]]
        left_Y = Y[X[best_split[0]] <= best_split[1]]
        right_X = X[X[best_split[0]] > best_split[1]]
        right_Y = Y[X[best_split[0]] > best_split[1]]
        return left_X, left_Y, right_X, right_Y
    
    def gini(self, Y):
        total = len(Y)
        if total == 0:
            return 0
        gini = 1
        for label in self.set_of_labels:
            probability = (Y == label).sum() / total
            gini -= probability ** 2
        return gini


"""
    On hérite seulement si :"A est un type de B
    Ex: Chien est un Animal

    DecisionTree : contient des Node , Il n’est pas un Node.
"""

#Une instance d’objet est un objet créé à partir d’une classe.

class Node:
    def __init__ (self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value


if __name__ == "__main__":

