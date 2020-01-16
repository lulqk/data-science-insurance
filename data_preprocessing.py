# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


def make_full_year(date):
    """ Funkcja przerabia date 15-07-85 na 15-07-1985,
    lub zwraca pd.NaT jeżeli argument nie jest tekstem. """
    if type(date) == str:
        full_year = '19' + date[6:]
        new_date = date[:6] + full_year
        return new_date
    else:
        return pd.NaT


def get_age(x):
    """Funkcja oblicza wiek na podstawie różnicy między rokiem urodzenia, a rokiem 1999"""
    return 1999 - x.year


def replace_french_months(data_series):
    """Funkcja podmienia francuskie nazwy miesięcy na format '-miesiac_num-'"""
    # Stworzenie dwóch list w celu podmiany francuskiej nazwy miesiąca
    french_months = ['janvier', 'février', 'mars', 'avril', 'mai', 'juin', 'juillet', 'aoűt', 'septembre', 'octobre',
                     'novembre', 'décembre']
    months = ['-01-', '-02-', '-03-', '-04-', '-05-', '-06-', '-07-', '-08-', '-09-', '-10-', '-11-', '-12-']
    for month in french_months:
        index = french_months.index(month)
        data_series = data_series.str.replace(month, months[index])

    return data_series


# Zbiór danych pochodzi z 1999 roku. - Analiza kolumn 'BIRTH' i AGE
def read_data_return_encoded():
    # Odczytanie danych z pliku
    data = pd.read_excel('dane.xls')

    # Podmiana kazdego miesiaca po kolei w kazdej kolumnie zawierajacej daty
    data.PLCYDATE = replace_french_months(data.PLCYDATE)
    data.INITDATE = replace_french_months(data.INITDATE)
    data.CLM_DATE = replace_french_months(data.CLM_DATE)
    data.BIRTH = replace_french_months(data.BIRTH)

    # Należy użyć funkcji make_full_year ponieważ daty występujące w plkiu są starsze niż 1970(Unix)
    # W przeciwnym razie funckja pd.to_datetime przerabia date 15-07-85 na 2085-07-15
    data.PLCYDATE = data.PLCYDATE.apply(make_full_year)
    data.INITDATE = data.INITDATE.apply(make_full_year)
    data.CLM_DATE = data.CLM_DATE.apply(make_full_year)
    data.BIRTH = data.BIRTH.apply(make_full_year)

    data.PLCYDATE = pd.to_datetime(data.PLCYDATE)
    data.INITDATE = pd.to_datetime(data.INITDATE)
    data.CLM_DATE = pd.to_datetime(data.CLM_DATE)
    data.BIRTH = pd.to_datetime(data.BIRTH)

    # Usunięcie znaku $ z wartości liczbowych
    data.BLUEBOOK = data.BLUEBOOK.str.replace('$', '')
    data.OLDCLAIM = data.OLDCLAIM.str.replace('$', '')
    data.CLM_AMT = data.CLM_AMT.str.replace('$', '')
    data.INCOME = data.INCOME.str.replace('$', '')
    data.HOME_VAL = data.HOME_VAL.str.replace('$', '')

    # Przekształcenie kolumn liczbowych z typem 'object' na kolumny typu 'int64' i 'float64'
    data.BLUEBOOK = pd.to_numeric(data.BLUEBOOK)
    data.OLDCLAIM = pd.to_numeric(data.OLDCLAIM)
    data.CLM_AMT = pd.to_numeric(data.CLM_AMT)
    data.INCOME = pd.to_numeric(data.INCOME)
    data.HOME_VAL = pd.to_numeric(data.HOME_VAL)

    # Przekształcenie kolumn z danymi 'yes/no' na kolumny typu 'bool'
    data.RED_CAR = data.RED_CAR.apply(str.lower)
    data.RED_CAR = data.RED_CAR.map({'yes': True, 'no': False})
    data.REVOKED = data.REVOKED.apply(str.lower)
    data.REVOKED = data.REVOKED.map({'yes': True, 'no': False})
    data.CLM_FLAG = data.CLM_FLAG.apply(str.lower)
    data.CLM_FLAG = data.CLM_FLAG.map({'yes': True, 'no': False})
    data.MARRIED = data.MARRIED.apply(str.lower)
    data.MARRIED = data.MARRIED.map({'yes': True, 'no': False})
    data.PARENT1 = data.PARENT1.apply(str.lower)
    data.PARENT1 = data.PARENT1.map({'yes': True, 'no': False})

    # Sprawdzenie ilości brakujących danych w każdej kolumnie
    print("Ilość brakujących danych w kolumnach")
    for column in data.columns:
        if data[column].isnull().values.any():
            print(column)
            print(data[column].isnull().sum())

    # Stworzenie macierzy korelacji
    corr_matrix = data.corr()
    # Stworzenie i zapisanie do pliku heatmapy korelacji
    mask = np.zeros_like(corr_matrix) # Maska - do przykrycia górnej cześći wykresu
    mask[np.triu_indices_from(mask)] = True
    f, ax = plt.subplots(figsize=(15,12))# figsize - określa rozmiar generowanego wykresu
    with sns.axes_style("white"):
        corr_plot = sns.heatmap(corr_matrix.round(2), mask=mask, square=True, annot=True, linewidths=.5, ax=ax)
    fig = corr_plot.get_figure()
    fig.savefig('correlation_matrix.png')# zapisanie wykresu do pliku

    # Usunięcie kolumny 'CLM_DATE' - 7556 brakujacych rekordów - tożsame z kolumną 'CLM_FLAG'
    data = data.drop(['CLM_DATE'], axis=1)

    # Usinięcie kolumny 'Age*Gender' - nieprzydatna
    data = data.drop(['AGE*GENDER'], axis=1)

    # Usinięcie kolumny 'YEARQTR' - nieprzydatna
    data = data.drop(['YEARQTR'], axis=1)

    # Usunięcie duplikatu z kolumny z numerami polis
    data = data.drop_duplicates(subset='POLICYNO')

    # Usunięcie kolumny 'POLICYNO' - numer nadawany dopiero po wykupieniu polisy
    data = data.drop(['POLICYNO'], axis=1)

    # Usunięcie kolumny 'PARENT1' - korelacja z 'HOMEKIDS' - 0,45
    data = data.drop(['PARENT1'], axis=1)

    # Usunięcie kolumny 'HOME_VAL' korelacja z 'INCOME' - 0,57
    data = data.drop(['HOME_VAL'], axis=1)

    # Usunięcie 'OLDCLAIM', wynika z 'CLM_FREQ' -  nie zajmujemy się wysokością CLM tylko ilością
    data = data.drop(['OLDCLAIM'], axis=1)

    # Usunięcie 'CLM_AMT', wynika z 'CLM_FLAG' - nie zajmujemy się wysokością CLM
    data = data.drop(['CLM_AMT'], axis=1)

    # Usunięcie kolumny 'ID' - dla jednego ID wystepują różne wartośći w zmiennych celu.
    data = data.drop(['ID'], axis=1)

    # Usunięcie kolumn 'PLCYDATE' i 'INITDATE' - różnica w latach jest w kolumnie 'RETAINED'
    data = data.drop(['PLCYDATE', 'INITDATE'], axis=1)

    # Usunięcie wiersza z wartoscia ujemna w kolumnie samehome
    data = data[data.SAMEHOME != -3]

    # Usunięcie kolumny 'INCOME' - brakuja dane a jest skorelowana z 'BLUEBOOK'(0,43)
    data = data.drop(['INCOME'], axis=1)

    # Dodanie kolumny z dokładnym wiekiem ubezpieczającego
    data['EXACT_AGE'] = data.BIRTH.apply(get_age)
    data = data.drop(['BIRTH'], axis=1)

    # Zakodowanie binarne zmiennych jakosciowych
    car_use = pd.get_dummies(data.CAR_USE.copy())
    car_type = pd.get_dummies(data.CAR_TYPE.copy())
    age = pd.get_dummies(data.AGE.copy())
    gender = pd.get_dummies(data.GENDER.copy())
    job_class = pd.get_dummies(data.JOBCLASS.copy())
    max_educ = pd.get_dummies(data.MAX_EDUC.copy())
    density = pd.get_dummies(data.DENSITY.copy())

    # Usuniecie kolumn typu object
    data = data.drop(['CAR_USE', 'CAR_TYPE', 'AGE', 'GENDER', 'JOBCLASS', 'MAX_EDUC', 'DENSITY'], axis=1)
    encoded_data = pd.concat([data, car_use, car_type, age, gender, job_class, max_educ, density], axis=1)

    # Przesuniecie kolumn 'CLM_FREQ' i 'CLM_FLAG' na koniec zestawu
    temp1 = encoded_data.pop('CLM_FREQ')
    temp2 = encoded_data.pop('CLM_FLAG')
    encoded_data['CLM_FREQ'] = temp1.copy()
    encoded_data['CLM_FLAG'] = temp2.copy()

    # Usunięcie kolumn z grupami wiekowymi - lepsze wyniki są otrzymywane z kolumną EXACT_AGE
    encoded_data = encoded_data.drop(['>60', '16-24', '25-40', '41-60'], axis=1)

    print("\nPozostawione kolumny:")
    print(encoded_data.columns)

    return encoded_data


def get_encoded_data_return_numerical(data):
    "Funckja przerabia dane na numeryczne, a następnie skaluje je przy użyciu StadanrdScaler"
    # Zmiana typu danych z BOOL na INT
    data.RED_CAR = data.RED_CAR.astype(int)
    data.REVOKED = data.REVOKED.astype(int)
    data.MARRIED = data.MARRIED.astype(int)

    # Usunięcie zmiennej celu 'CLM_FLAG'
    data = data.drop(['CLM_FLAG'], axis=1)

    # Przeskalowanie zmiennych liczbowych
    columns_to_scale = ['KIDSDRIV', 'TRAVTIME', 'BLUEBOOK',
                        'RETAINED', 'NPOLICY', 'MVR_PTS',
                        'HOMEKIDS', 'YOJ', 'SAMEHOME', 'EXACT_AGE']
    std_scl = StandardScaler()
    std_data = data.copy()
    std_data[columns_to_scale] = std_scl.fit_transform(std_data[columns_to_scale])

    return std_data


def run_pca(data):
    """Funkcja przeprowadza PCA na danym zbiorze danych.
    Dane muszą być w formacie numerycznym."""
    n = 20
    pca = PCA(n_components=n)
    principal_components = pca.fit_transform(data)
    exp_comp = pca.explained_variance_ratio_
    exp_com_cum = np.cumsum(exp_comp)
    
    headers = []
    for i in range(n):
        headers.append('PrincipalComponent '+str(i))
        
    for i in range(len(exp_comp)):
        print(i, exp_comp[i], exp_com_cum[i])
        
    principal_df = pd.DataFrame(data=principal_components, columns=headers)
    
    return principal_df
