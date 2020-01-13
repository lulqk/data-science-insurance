# -*- coding: utf-8 -*-

# Import bibliotek
import pandas as pd
import datetime
from handle_classification_models import clm_flag_classification
from handle_regression_models import clm_freq_regression
from data_preprocessing import read_data_return_encoded, get_encoded_data_return_numerical, run_pca
from evaluation import print_classification_results, print_regression_results

def main():
    start_time = datetime.datetime.now()
    encoded_data = read_data_return_encoded()

    # Podzbiór 1: Ze zbioru danych usuwamy wszystkie wiersze w których brakuje wartosci ('YOJ', 'SAMEHOME')
    # pozostaje 9134 z 10294.
    encoded_data_no_na = encoded_data.dropna()
    # Podzbiór 2: Ze zbioru usuwamy kolumny 'YOJ' i 'SAMEHOME' - brakuje w nich danych.
    encoded_data_no_yoj_samehome = encoded_data.drop(['YOJ', 'SAMEHOME'], axis=1)

    # Wyniki dla klasyfikacji przy 2 podejściach do brakujących wartości w zbiorze
    results = clm_flag_classification(encoded_data_no_na)
    print("\n\n\nKlasyfikacja\n")
    print("\nDane bez wierszy zawierających NaN.")
    print_classification_results(results)


    results2 = clm_flag_classification(encoded_data_no_yoj_samehome)
    print("\nDane bez kolumn 'YOJ' i 'SAMEHOME', w których brakowało wartości.")
    print_classification_results(results2)

    # Przekształcenie wszystkich danych jakościowych na wartości numeryczne (dummy variables)
    numerical_data = get_encoded_data_return_numerical(encoded_data.copy())

    numerical_data_drop_na = numerical_data.dropna()
    numerical_data_drop_na = numerical_data_drop_na.reset_index(drop=True)
    numerical_data_no_yoj_samehome = numerical_data.dropna(axis=1)
    numerical_data_no_yoj_samehome = numerical_data_no_yoj_samehome.reset_index(drop=True)

    # Wyniki dla regresji przy 2 podejściach do brakujących wartości w zbiorze
    reg_results = clm_freq_regression(numerical_data_drop_na)
    print("\n\n\nRegresja\n")
    print("\nDane bez wierszy zawierających NaN.")
    print_regression_results(reg_results)


    reg_results2 = clm_freq_regression(numerical_data_no_yoj_samehome)
    print("\nDane bez kolumn 'YOJ' i 'SAMEHOME'")
    print_regression_results(reg_results2)


    # Zastosowanie PCA na zbiorze w formie numerycznej.
    # Pozostawione zostaje 20 komponentów, które opisują 92% danych
    labels1 = numerical_data_drop_na['CLM_FREQ'].copy()
    labels2 = numerical_data_no_yoj_samehome['CLM_FREQ'].copy()
    data1 = numerical_data_drop_na.drop(['CLM_FREQ'], axis=1)
    data2 = numerical_data_no_yoj_samehome.drop(['CLM_FREQ'], axis=1)
    data1_pca = run_pca(data1)
    data2_pca = run_pca(data2)

    data1_df = pd.concat([data1_pca, labels1], axis=1)
    data2_df = pd.concat([data2_pca, labels2], axis=1)

    # Wyniki dla regresji z zastosowaniem PCA przy 2 podejściach do brakujących wartości w zbiorze
    reg_results_pca = clm_freq_regression(data1_df, pca=True)
    print("\n\n\nRegresja z PCA\n")
    print("\nDane bez wierszy zawierających NaN.")
    print_regression_results(reg_results_pca)

    reg_results_pca2 = clm_freq_regression(data2_df, pca=True)
    print("\nDane bez kolumn 'YOJ' i 'SAMEHOME'")
    print_regression_results(reg_results_pca2)

    # Pomiar czasu
    end_time = datetime.datetime.now()
    total_time = end_time-start_time

    print("time elapsed seconds: ", total_time.seconds)
    print("minutes = ", total_time.seconds/60)

if __name__ == '__main__':
    main()