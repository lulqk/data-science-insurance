import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix, precision_score, recall_score
from sklearn.linear_model import SGDClassifier


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
    export_graphviz(
        model,
        out_file="C:/Users/grulk/Documents/data_science/ubezpieczenia/" + filename + ".dot",
        feature_names=col_names,
        class_names=['True', 'False'],
        rounded=True,
        filled=True)


def decision_tree_grid(X_train_set, y_train_set, X_test_set, y_test_set):
    """Trening i test modelu drzewa decyzyjnego. Funkcja generuje także wykres modelu drzewa"""

    # Użycie klasy GridSearchCV do znalezienia najbardziej optymalnego drzewa
    parameters_tree = {'max_depth': range(3, 20)}
    tree_clf = GridSearchCV(DecisionTreeClassifier(), parameters_tree, n_jobs=-1)
    tree_clf.fit(X=X_train_set, y=y_train_set)
    tree_model = tree_clf.best_estimator_
    print("\nNajlepszy wynik drzewa decyzyjnego: ", tree_clf.best_score_)
    print("\nParametry najlepszego drzewa: ", tree_clf.best_params_)

    # Narysuj drzewo
    draw_decision_tree(tree_model, X_train_set.columns, "drzewo_grid")

    # Cechy od najbardziej do najmniej znaczących
    print("\nModel: DecisionTreeClasiffier\nDane bez wierszy z NaN.\n")
    features_sorted = sorted(zip(tree_model.feature_importances_, X_train_set.columns, ), reverse=True)
    features_frame = pd.DataFrame(features_sorted, columns=['Importance', 'Feature'])
    print("Cechy drzewa decyzyjnego:\n", features_frame)

    # kroswalidacja predykcji drzewa i stworzenie macierzy pomyłek
    y_train_tree_pred = cross_val_predict(tree_model, X_train_set, y_train_set, cv=3)
    conf_matrix = confusion_matrix(y_train_set, y_train_tree_pred)
    print("\nMacierz pomyłek drzewa dla zbioru treningowego:\n", conf_matrix)

    # Klasyfikacja na zbiorze testowym i stworzenie macierzy pomyłek
    tree_final_classifications = tree_model.predict(X_test_set)
    final_tree_conf_matrix = confusion_matrix(y_test_set, tree_final_classifications)

    # Obliczenie precyzji i pełności
    tree_precision = precision_score(y_test_set, tree_final_classifications)
    tree_recall = recall_score(y_test_set, tree_final_classifications)

    # Stworzenie obiektu z wynikami modelu
    tree_result = ResultDataClassifiers(model_name='DecisionTree',
                                        model=tree_model,
                                        precision=tree_precision,
                                        recall=tree_recall,
                                        conf_matrix=final_tree_conf_matrix)

    return tree_result


def random_forest_grid(X_train_set, y_train_set, X_test_set, y_test_set):
    """Trening i test modelu lasu losowego."""

    # Użycie klasy GridSearchCV do znalezienia najbardziej optymalnego lasu losowego
    parameters_rand = {'max_leaf_nodes': range(10, 40)}
    rand_clf = GridSearchCV(RandomForestClassifier(n_estimators=1000, random_state=42),
                            parameters_rand, n_jobs=-1)
    rand_clf.fit(X=X_train_set, y=y_train_set)
    randforest_model = rand_clf.best_estimator_
    print("\nNajlepszy wynik lasu losowego: ", rand_clf.best_score_)
    print("\nParametry najlepszego lasu losowego: ", rand_clf.best_params_)

    # Cechy od najbardziej do najmniej znaczących
    print("\nModel: RandomForestClasiffier\nDane bez wierszy z NaN.\n")
    features_sorted = sorted(zip(randforest_model.feature_importances_, X_train_set.columns, ), reverse=True)
    features_frame = pd.DataFrame(features_sorted, columns=['Importance', 'Feature'])
    print("Cechy lasu losowego:\n", features_frame)
    # kroswalidacja predykcji lasu losowego i stworzenie macierzy pomyłek
    y_train_randforest_pred = cross_val_predict(randforest_model, X_train_set, y_train_set, cv=3)
    conf_matrix = confusion_matrix(y_train_set, y_train_randforest_pred)
    print("\nMacierz pomyłek lasu losowego dla zbioru treningowego:\n", conf_matrix)

    # Klasyfikacja na zbiorze testowym i stworzenie macierzy pomyłek
    randforest_final_classifications = randforest_model.predict(X_test_set)
    final_randforest_conf_matrix = confusion_matrix(y_test_set, randforest_final_classifications)

    # Obliczenie precyzji i pełności
    randforest_precision = precision_score(y_test_set, randforest_final_classifications)
    randforest_recall = recall_score(y_test_set, randforest_final_classifications)

    # Stworzenie obiektu z wynikami modelu
    randforest_result = ResultDataClassifiers(model_name='RandomForest',
                                              model=randforest_model,
                                              precision=randforest_precision,
                                              recall=randforest_recall,
                                              conf_matrix=final_randforest_conf_matrix)

    return randforest_result


def sgd(X_train_set, y_train_set, X_test_set, y_test_set):
    """Trening i test modelu stochastycznego spadku wzdłóż gradientu"""

    # Stworzenie klasyfikatora stochstycznego spadku gradientu
    sgd_clf = SGDClassifier(random_state=42)

    # Trening klasyfikatora
    sgd_clf.fit(X=X_train_set, y=y_train_set)

    # kroswalidacja predykcji klasyfikatora SGD i stworzenie macierzy pomyłek
    y_train_sgd_pred = cross_val_predict(sgd_clf, X_train_set, y_train_set, cv=3)
    conf_matrix = confusion_matrix(y_train_set, y_train_sgd_pred)
    print("\nMacierz pomyłek klasyfikatora SGD dla zbioru treningowego:\n", conf_matrix)

    # Klasyfikacja na zbiorze testowym i stworzenie macierzy pomyłek
    sgd_final_classifications = sgd_clf.predict(X_test_set)
    final_sgd_conf_matrix = confusion_matrix(y_test_set, sgd_final_classifications)

    # Obliczenie precyzji i pełności
    sgd_precision = precision_score(y_test_set, sgd_final_classifications)
    sgd_recall = recall_score(y_test_set, sgd_final_classifications)

    # Stworzenie obiektu z wynikami modelu
    sgd_result = ResultDataClassifiers(model_name='SGD',
                                       model=sgd_clf,
                                       precision=sgd_precision,
                                       recall=sgd_recall,
                                       conf_matrix=final_sgd_conf_matrix)

    return sgd_result