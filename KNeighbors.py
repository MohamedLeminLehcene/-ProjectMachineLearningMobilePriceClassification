#Etapes demarche K-Nearest Neighbors (KNN)
#1- Calculer la distance entre toutes les données d’entraînement et le point de test
#2- Trouvez les k voisins les plus proches en triant ces distances par paires
#3- Classifier le point sur la base d’un vote majoritaire

"""
1 - Tout d'abord, on doit importer les bibliothèques nécessaires :
 numpy pour les calculs numériques et Counter pour le comptage des occurrences.
"""
import numpy as np
from collections import Counter

class KNNClassifier:
    """
        La classe KNNClassifier est définie avec une méthode __init__ pour initialiser
        le nombre de voisins k, ainsi que les attributs x_train et y_train qui seront
        utilisés pour stocker les données d'entraînement.
"""
    def __init__(self, k):
        self.k = k
        self.x_train = None
        self.y_train = None

    """
    La méthode fit est utilisée pour fournir les données d'entraînement au classifieur. 
    Elle prend en entrée x_train et y_train,qui correspondent respectivement aux 
    caractéristiques et aux étiquettes de l'ensemble d'entraînement.
    """
    def fit(self, x_train, y_train):
        self.x_train = x_train
        self.y_train = y_train

    """
    La méthode euclidean_distance calcule la distance euclidienne entre deux points x1
    et x2 en utilisant la formule mathématique de la distance euclidienne.
    """
    def euclidean_distance(self, x1, x2):
        return np.sqrt(np.sum((x1 - x2) ** 2))
    """
    La méthode find_nearest_neighbors est utilisée pour trouver les k voisins les plus 
    proches d'un point de test x_test_point. Elle calcule les distances entre 
    x_test_point et chaque point de l'ensemble d'entraînement, puis trie les distances
      de manière croissante. Les k voisins les plus proches sont extraits à partir des 
      indices triés et renvoyés sous forme de liste.
    """
    def find_nearest_neighbors(self, x_test_point):
        distances = []
        for row in range(len(self.x_train)):
            current_train_point = self.x_train[row]
            current_distance = self.euclidean_distance(current_train_point, x_test_point)
            distances.append((current_distance, row))

        distances.sort(key=lambda x: x[0])
        nearest_neighbors = [self.y_train[index] for (_, index) in distances[:self.k]]
        return nearest_neighbors

    """
    La méthode majority_vote effectue un vote à la majorité parmi les voisins les plus
    proches nearest_neighbors.Elle compte le nombre d'occurrences de chaque classe
    dans nearest_neighbors à l'aide de Counter et renvoie la classe la plus fréquente.
    """
    def majority_vote(self, nearest_neighbors):
        counter_vote = Counter(nearest_neighbors)
        y_pred = counter_vote.most_common(1)[0][0]
        return y_pred

    """
    La méthode predict est utilisée pour prédire les étiquettes des points de 
    test x_test. Pour chaque point de test, elle trouve les k voisins
    les plus proches à l'aide de find_nearest_neighbors et effectue un vote à 
    la majorité avec majority_vote. Les prédictions sont stockées 
    dans une liste y_pred qui est renvoyée à la fin.
    """
    def predict(self, x_test):
        y_pred = []
        for x_test_point in x_test:
            nearest_neighbors = self.find_nearest_neighbors(x_test_point)
            y_pred_point = self.majority_vote(nearest_neighbors)
            y_pred.append(y_pred_point)
        return y_pred

"""
En utilisant cette classe KNNClassifier, vous pouvez entraîner le modèle en fournissant
les données d'entraînement avec la méthode fit, 
puis effectuer des prédictions sur de nouveaux points avec la méthode predict
"""