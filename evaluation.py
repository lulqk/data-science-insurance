# -*- coding: utf-8 -*-
def print_classification_results(results):
    """Wyświetlenie na ekranie infomacji o wynikach danego klasyfikatora """
    for name, model in results.items():
        print("\n" + name + "\n")
        print("\tPrecyzja: " + str(results[name].precision))
        print("\tPełność: " + str(results[name].recall))
        print("\tMacierz pomyłek:\n",results[name].conf_matrix)


def print_regression_results(results):
    """Wyświetlenie na ekranie infomacji o wynikach danego regresora """
    for name, model in results.items():
        print("\n" + model.model_name + ":\n")
        print("\tBłąd RMSE trening: " + str(model.rmse_train))
        print("\tBłąd RMSE test: " + str(model.rmse_test))
        print("\tWyniki kroswalidacji:\n")
        print("\t\tŚrednia: " + str(model.cvs_mean))
        print("\t\tOdchylenie standardowe: " + str(model.cvs_std))
        print("\t\tWyniki: " + str(model.cvs_scores))
