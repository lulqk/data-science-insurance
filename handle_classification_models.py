# -*- coding: utf-8 -*-
import datetime

from sklearn.model_selection import train_test_split
from sklearn.tree import export_graphviz

from classification_models import decision_tree_grid, random_forest_grid, sgd

class ResultDataClassifiers:
    """Klasa w której zapisuje wynik każdego algorytmu"""

    def __init__(self, model_name, model, precision, recall, conf_matrix):
        self.model_name = model_name
        self.model = model
        self.precision = precision
        self.recall = recall
        self.conf_matrix = conf_matrix


def draw_decision_tree(model, col_names, filename):
    """Funckja tworzy schemat drzewa dezyjnego w formacie '.dot'"""
    date = datetime.datetime.now().strftime('%d/%m/%y_%H%M')
    export_graphviz(
        model,
        out_file="/drzewa/" + filename + ".dot",
        feature_names=col_names,
        class_names=['True', 'False'],
        rounded=True,
        filled=True)


def clm_flag_classification(data):
    """    Klasyfikacja CLM_FLAG - True/False.    """
    # Usunięcie kolumny 'CLM_FREQ'
    #data_no_freq = data.drop(['CLM_FREQ'], axis=1)
    # Stworzenie zestawów uczącego i testowego
    train_set, test_set = train_test_split(data, test_size=0.2, random_state=42)

    X_train_set = train_set.loc[:, :'Urban']
    y_train_set = train_set['CLM_FLAG']
    X_test_set = test_set.loc[:, :'Urban']
    y_test_set = test_set['CLM_FLAG']

    # Trening i test wybranych modeli
    tree_result = decision_tree_grid(X_train_set, y_train_set, X_test_set, y_test_set)
    randforest_result = random_forest_grid(X_train_set, y_train_set, X_test_set, y_test_set)
    sgd_result = sgd(X_train_set, y_train_set, X_test_set, y_test_set)

    # Zapisanie wyników w słowniku
    final_scores = {"DecisionTree": tree_result,
                    "RandomForest": randforest_result,
                    "SGD": sgd_result}

    return final_scores
