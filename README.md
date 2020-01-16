#### data-science-insurance
# Projekt - Stworzenie modelu na podstawie danych ubezpieczeniowych
## Cele:
- Predykcja ilości rozczeń z danej polisy
- Klasykacja polis - czy z danej polisy będzie rozczenie

Dane pochodzą z francuskiej ubezpieczalni z roku 1999.

Plik z danymi: dane.xls


## Przygotowanie i analiza danych:
Opis kolumn:
1. ID - ”integer” ID
2. KIDSDRIV - ”integer”, the number of driving children
3. PLCYDATE - ”Date”, policy effective date
4. TRAVTIME - ”integer”, the distance to work
5. CAR USE - ”factor”, the primary use of the vehicle: ”Commercial”, ”Private”.
6. POLICYNO - ”character”, the policy number
7. BLUEBOOK - ”integer”, the value of the vehicle
8. INITDATE - ”Date”, policy initiation date
9. RETAINED - ”integer”, the number of years as a customer
10. NPOLICY - ”integer”, the number of policies
11. CAR TYPE - ”factor”, the type of the car: ”Panel Truck”, ”Pickup”, ”Sedan”, ”Sports Car”, ”SUV”, ”Van”.
12. RED CAR - ”factor”, whether the color of the car is red: ”no”, ”yes”.
13. OLDCLAIM - ”integer”, the total claim amount in the past years
14. CLM FREQ - ”integer”, the number of claims in the past 5 years
15. REVOKED - ”factor”, whether the dirver’s license was invoked in the past7 years: ”No”, ”Yes”
16. MVR PTS - ”integer”, MVR violation records (MOTOR VEHICLE VIOLATIONS)
17. CLM AMT - ”integer”, the claim amount in the current insured period
18. CLM DATE - ”date” claim date
19. CLM FLAG - ”factor”, whether a claim is
20. AGE - ”integer”, the age of the driver
21. AGE*GENDER -”integer”, the age of the driver merged with gender
22. HOMEKIDS - ”integer”, the number of children
23. YOJ - ”integer”, years at current job
24. INCOME - ”integer”, annual income
25. GENDER - ”factor”, the gender of the driver: ”F”, ”M”
26. MARRIED - ”factor”, married or not: ”No”, ”Yes”

Proces przygotowania danych krok po kroku znajduje się w pliku data_preprocessing.py w funkcjach:
* read_data_return_endoced
* get_encoded_data_return_numerical

W trakcie obróbki danych stworzyłem macierz korelacji, której wyskres został wygenerowany przy użycniu pakietu seaborn.
![corr_coef](correlation_matrix.png)



Ilość brakujących wierszy w kolumnach:

* CLM_DATE
    * 7556
* YOJ
    * 548
* INCOME
    * 569
* JOBCLASS
    * 665
* HOME_VAL
    * 574
* SAMEHOME 
    * 639

Pozostawione kolumny:
* KIDSDRIV
* TRAVTIME
* BLUEBOOK
* RETAINED
* NPOLICY
* RED_CAR
* REVOKED
* MVR_PTS
* HOMEKIDS
* YOJ
* MARRIED
* SAMEHOME
* EXACT_AGE
* Commercial
* Private
* Panel Truck
* Pickup
* SUV
* Sedan
* Sports Car
* Van
* F
* M
* Blue Collar
* Clerical
* Doctor
* Home Maker
* Lawyer
* Manager
* Professional
* Student
* <High School
* Bachelors
* High School
* Masters
* PhD
* Highly Rural
* Highly Urban
* Rural
* Urban
* CLM_FREQ
* CLM_FLAG


#Wyniki analizy danych
Po analizie postanowiono:
* Zbinaryzować dane jakościowe
* Usunąć daty - jako, że nie wnoszą nic do predykcji
* Zastosować dwa podejścia do brakujących danych
    * Usunąć wiersze w których wystepowały wartości NaN (około 10% rekordów)
    * Usunąć kolumny w których występowały wartości NaN (kolumny YOJ oraz SAMEHOME)
* Stworzyć kolumnę EXACT_AGE, a następnie usunąć kolumy z grupami wiekowymi


#Tworzenie modeli
Modele zostały zaczerpnięte z biblioteki scikit-learn.

##Modele klasyfikujące:
Modele znajdują się w pliku classification_models.py, a funkcja do obsługi modeli i zbierania ich wyników znajduje się w pliku handle_classification_models.py

##Modele regresji:
Modele oraz funkcja do zebrania ich wyników znajdują się w pliku handle_regression_models.py



#Wyniki poszczególnych modeli:
##Klasyfikacja - dane trenigowe:
###Dane bez wierszy z NaN.

###Model: DecisionTreeClasiffier

Najlepszy wynik drzewa decyzyjnego (accuracy):  0.7678938004653072

Parametry najlepszego drzewa:  {'max_depth': 7}
![drzewo](drzewo_gridsearch.png)
Cechy drzewa decyzyjnego:

No. | Importance | Feature
--- | ---------- | -------
0 | 0.301011 | Highly Urban
1 | 0.113122 | CLM_FREQ
2 | 0.085123 | BLUEBOOK
3 | 0.076676 | EXACT_AGE
4 | 0.053267 | Private
5 | 0.051481 | MVR_PTS
6 | 0.046314 | SAMEHOME
7 | 0.040829 | KIDSDRIV
8 | 0.039799 | TRAVTIME
9 | 0.032534 | RETAINED
10 | 0.030602 | MARRIED
11 | 0.020281 | YOJ
12 | 0.018918 | Sedan
13 | 0.018842 | REVOKED
14 | 0.018149 | Urban
15 | 0.008563 | Commercial
16 | 0.008362 | NPOLICY
17 | 0.006513 | Blue Collar
18 | 0.005622 | RED_CAR
19 | 0.005055 | Rural
20 | 0.004867 | High School
21 | 0.004488 | PhD
22 | 0.003056 | HOMEKIDS
23 | 0.002243 | Sports Car
24 | 0.002072 | Professional
25 | 0.001376 | Pickup
26 | 0.000834 | Student
27 | 0.000000 | Van
28 | 0.000000 | SUV
29 | 0.000000 | Panel Truck
30 | 0.000000 | Masters
31 | 0.000000 | Manager
32 | 0.000000 | M
33 | 0.000000 | Lawyer
34 | 0.000000 | Home Maker
35 | 0.000000 | Highly Rural
36 | 0.000000 | F
37 | 0.000000 | Doctor
38 | 0.000000 | Clerical
39 | 0.000000 | Bachelors
40 | 0.000000 | <High School

Macierz pomyłek drzewa dla zbioru treningowego:

X | Positive | Negative
--- | ---------- | -------
Positive | 4834 | 502
Negative | 1196 | 775


###Model: RandomForestClasiffier

Najlepszy wynik lasu losowego (accuracy):  0.7817161625838237

Parametry najlepszego lasu losowego:  {'max_leaf_nodes': 39}

Macierz pomyłek lasu losowego dla zbioru treningowego:

X | Positive | Negative
--- | ---------- | -------
Positive | 5135 | 201
Negative | 1394 | 577

###Model: SGDClassifier

Macierz pomyłek klasyfikatora SGD dla zbioru treningowego:

X | Positive | Negative
--- | ---------- | -------
Positive | 5191 | 145
Negative | 1901 | 70


##Klasyfikacja dane testowe:

DecisionTree

	Precyzja: 0.5655172413793104
	Pełność: 0.35807860262008734
	Macierz pomyłek:
	
X | Positive | Negative
--- | ---------- | -------
Positive | 1243 | 126
Negative | 294 | 164

RandomForest

	Precyzja: 0.7678571428571429
	Pełność: 0.2816593886462882
	Macierz pomyłek:
	
X | Positive | Negative
--- | ---------- | -------
Positive | 1330 | 39
Negative | 329 | 129

SGD

	Precyzja: 0.25068418171866447
	Pełność: 1.0
	Macierz pomyłek:

X | Positive | Negative
--- | ---------- | -------
Positive | 0 | 1369
Negative | 0 | 458
 

###Dane bez kolumn 'YOJ' i 'SAMEHOME', w których brakowało wartości.

###Model: DecisionTreeClasiffier
Najlepszy wynik drzewa decyzyjnego (accuracy):  0.7672131147540984

Parametry najlepszego drzewa:  {'max_depth': 8}

Macierz pomyłek drzewa dla zbioru treningowego:

X | Positive | Negative
--- | ---------- | -------
Positive | 5478 | 594
Negative | 1316 | 847

###Model: RandomForestClasiffier

Najlepszy wynik lasu losowego:  0.7857923497267759

Parametry najlepszego lasu losowego:  {'max_leaf_nodes': 39}

Macierz pomyłek lasu losowego dla zbioru treningowego:

X | Positive | Negative
--- | ---------- | -------
Positive | 5880 | 192
Negative | 1572 | 591

###Model: SGDCClassifier

Macierz pomyłek klasyfikatora SGD dla zbioru treningowego:


X | Positive | Negative
--- | ---------- | -------
Positive | 541 | 621
Negative | 1799 | 364

##Klasyfikacja - dane testowe:

DecisionTree

	Precyzja: 0.6311688311688312
	Pełność: 0.421875
	Macierz pomyłek:
	
X | Positive | Negative
--- | ---------- | -------
Positive | 1341 | 142
Negative | 333 | 243

RandomForest

	Precyzja: 0.8199052132701422
	Pełność: 0.3003472222222222
	Macierz pomyłek:

X | Positive | Negative
--- | ---------- | -------
Positive | 1445 | 38
Negative | 403 | 173

SGD

	Precyzja: 0.2797474502185527
	Pełność: 1.0
	Macierz pomyłek:

X | Positive | Negative
--- | ---------- | -------
Positive | 0 | 1483
Negative | 0 | 576


#Regresja

##Dane bez wierszy zawierających NaN.

Linear Regressor:

	Błąd RMSE trening: 1.022717908817294
	Błąd RMSE test: 1.006657105386976
	Wyniki kroswalidacji:

		Średnia: 1.0277301540557873
		Odchylenie standardowe: 0.02872217010004909
		Wyniki: [1.00223869 1.0467991  1.01711326 1.03854124 1.07493905 1.00177982 0.99567861 1.05444794 0.98885    1.05691382]

Decision Tree Regressor:

	Błąd RMSE trening: 0.0
	Błąd RMSE test: 1.5095028306830613
	Wyniki kroswalidacji:

		Średnia: 1.4783978420242552
		Odchylenie standardowe: 0.03395262860020954
		Wyniki: [1.4748229  1.48683231 1.49188411 1.48268622 1.54016006 1.40060569 1.45662316 1.48416297 1.50342076 1.46278024]

Random Forest Regressor:

	Błąd RMSE trening: 0.4596917943340093
	Błąd RMSE test: 1.0781199553874825
	Wyniki kroswalidacji:

		Średnia: 1.0858471189261882
		Odchylenie standardowe: 0.03162523560820953
		Wyniki: [1.07357898 1.12245456 1.06935935 1.08659284 1.133165   1.06885392 1.05295493 1.10610386 1.02869781 1.11670994]

RF GSCV Regressor:

	Błąd RMSE trening: 0.38858632169719337
	Błąd RMSE test: 1.0151255109372757
	Wyniki kroswalidacji:

		Średnia: 1.0334600606660385
		Odchylenie standardowe: 0.028838748167535955
		Wyniki: [1.00480705 1.06165507 1.01596281 1.03413055 1.08582718 1.00786239 1.00629707 1.05362525 1.00181048 1.06262276]

Lasso Regressor:

	Błąd RMSE trening: 1.1580600212903933
	Błąd RMSE test: 1.1378341785307031
	Wyniki kroswalidacji:

		Średnia: 1.1579271542511755
		Odchylenie standardowe: 0.021021140125919748
		Wyniki: [1.12730351 1.18433321 1.17040813 1.16363312 1.18846984 1.12514572 1.14808935 1.16205533 1.13981642 1.17001693]

LassoCV Regressor:

	Błąd RMSE trening: 1.0233367384222956
	Błąd RMSE test: 1.0070417247373704
	Wyniki kroswalidacji:

		Średnia: 1.027822849566188
		Odchylenie standardowe: 0.02864026252894873
		Wyniki: [1.00306095 1.04856703 1.01720328 1.0383935  1.07637231 1.0033934 0.99504489 1.0526503  0.98847896 1.05506386]

## Dane bez kolumn 'YOJ' i 'SAMEHOME'

Linear Regressor:

	Błąd RMSE trening: 1.0255940452298504
	Błąd RMSE test: 1.0135169017547263
	Wyniki kroswalidacji:

		Średnia: 1.0299041241221847
		Odchylenie standardowe: 0.016740422828721513
		Wyniki: [1.04733132 1.04493961 1.01980629 1.03676483 1.04269738 1.00003345 1.04160109 1.00377251 1.04212416 1.01997061]

Decision Tree Regressor:

	Błąd RMSE trening: 0.0
	Błąd RMSE test: 1.510847936146563
	Wyniki kroswalidacji:

		Średnia: 1.4815173784756221
		Odchylenie standardowe: 0.05418030080199977
		Wyniki: [1.52527254 1.49716561 1.50806891 1.37196266 1.46313933 1.41206397 1.47765865 1.48667635 1.5767139  1.49645186]

Random Forest Regressor:

	Błąd RMSE trening: 0.4607575399549957
	Błąd RMSE test: 1.092995047505162
	Wyniki kroswalidacji:

		Średnia: 1.094207061757508
		Odchylenie standardowe: 0.024610483621689584
		Wyniki: [1.1346164  1.0715371  1.08476318 1.09015898 1.13812996 1.06590255 1.1046131  1.07292992 1.1050805  1.07433893]

RF GSCV Regressor:

	Błąd RMSE trening: 0.3881256627107319
	Błąd RMSE test: 1.0309895188858385
	Wyniki kroswalidacji:

		Średnia: 1.0360688581721236
		Odchylenie standardowe: 0.024272554585187317
		Wyniki: [1.07462276 1.02325171 1.03507264 1.02561798 1.07297987 1.0040624 1.03928649 1.00576884 1.05981549 1.02021041]

Lasso Regressor:

	Błąd RMSE trening: 1.156328659694595
	Błąd RMSE test: 1.1430864849765685
	Wyniki kroswalidacji:

		Średnia: 1.156255019501933
		Odchylenie standardowe: 0.017328587465414772
		Wyniki: [1.18718857 1.15645518 1.15094435 1.17835695 1.16493817 1.13283149 1.14973524 1.12737893 1.15373185 1.16098947]

LassoCV Regressor:

	Błąd RMSE trening: 1.026315982613284
	Błąd RMSE test: 1.012813647788806
	Wyniki kroswalidacji:

		Średnia: 1.0297920096966464
		Odchylenie standardowe: 0.016113305826359958
		Wyniki: [1.04546335 1.04277754 1.01892075 1.03641699 1.04234858 1.00100464 1.04406138 1.00563427 1.04171531 1.01957729]


# PCA dla zbioru danych - bez wierszy z wartością NaN

Lp. | Explained Component | Cummulative
--- | ---------- | -------
0 | 0.1341883337131534 | 0.1341883337131534
1 | 0.0916728453085964 | 0.2258611790217498
2 | 0.07227392908750346 | 0.29813510810925326
3 | 0.07126184411033083 | 0.3693969522195841
4 | 0.07011821594515764 | 0.43951516816474173
5 | 0.06777241612858925 | 0.507287584293331
6 | 0.06613225002776521 | 0.5734198343210962
7 | 0.06275143387151884 | 0.636171268192615
8 | 0.05582375307235175 | 0.6919950212649668
9 | 0.04938163069306438 | 0.7413766519580312
10 | 0.030944543404201257 | 0.7723211953622324
11 | 0.027335482559857703 | 0.7996566779220902
12 | 0.02601056777543819 | 0.8256672456975284
13 | 0.019864662744834453 | 0.8455319084423628
14 | 0.015745317259301936 | 0.8612772257016648
15 | 0.015145800761361867 | 0.8764230264630266
16 | 0.013477977822670914 | 0.8899010042856975
17 | 0.012792470184620777 | 0.9026934744703183
18 | 0.01213536158992159 | 0.9148288360602399
19 | 0.010703091058646225 | 0.9255319271188861


#Regresja z PCA

##Dane bez wierszy zawierających NaN.

Linear Regressor:

	Błąd RMSE trening: 1.0295569107606688
	Błąd RMSE test: 1.0140405677688775
	Wyniki kroswalidacji:

		Średnia: 1.0321425733967158
		Odchylenie standardowe: 0.028084329067678142
		Wyniki: [1.00516825 1.05455291 1.02297403 1.04358278 1.07894659 1.0092137 1.00049159 1.05421541 0.99323797 1.0590425 ]

Decision Tree Regressor:

	Błąd RMSE trening: 0.0
	Błąd RMSE test: 1.4634761535834055
	Wyniki kroswalidacji:

		Średnia: 1.4837794632656847
		Odchylenie standardowe: 0.054861205642766336
		Wyniki: [1.47528661 1.58221995 1.45050583 1.49004907 1.52229217 1.46738365 1.36499124 1.48831062 1.46043716 1.53631832]

Random Forest Regressor:

	Błąd RMSE trening: 0.4617488370360819
	Błąd RMSE test: 1.0817893506749439
	Wyniki kroswalidacji:

		Średnia: 1.0948642084747593
		Odchylenie standardowe: 0.033041231560901815
		Wyniki: [1.04948036 1.12027088 1.05503159 1.108591   1.15946707 1.07259099 1.07022889 1.12119677 1.08064542 1.11113912]

RF GSCV Regressor:

	Błąd RMSE trening: 0.3913129671213397
	Błąd RMSE test: 1.0288966409488778
	Wyniki kroswalidacji:

		Średnia: 1.0432480236777488
		Odchylenie standardowe: 0.02804147796716817
		Wyniki: [1.00474599 1.07470112 1.02008516 1.04924577 1.08429177 1.01881761 1.02203808 1.07172585 1.01814874 1.06868015]

Lasso Regressor:

	Błąd RMSE trening: 1.1580600212903933
	Błąd RMSE test: 1.1378341785307031
	Wyniki kroswalidacji:

		Średnia: 1.1579271542511755
		Odchylenie standardowe: 0.021021140125919748
		Wyniki: [1.12730351 1.18433321 1.17040813 1.16363312 1.18846984 1.12514572 1.14808935 1.16205533 1.13981642 1.17001693]

LassoCV Regressor:

	Błąd RMSE trening: 1.0295764791045192
	Błąd RMSE test: 1.0139578528108033
	Wyniki kroswalidacji:

		Średnia: 1.0321354643873826
		Odchylenie standardowe: 0.02815342403665364
		Wyniki: [1.00503369 1.05448074 1.02284862 1.04403261 1.07889511 1.00935552 1.00043584 1.0550171  0.99287087 1.05838454]

##Dane bez kolumn 'YOJ' i 'SAMEHOME'

Linear Regressor:

	Błąd RMSE trening: 1.0322802071259918
	Błąd RMSE test: 1.0152589230160032
	Wyniki kroswalidacji:

		Średnia: 1.0348728512192076
		Odchylenie standardowe: 0.015932141985127
		Wyniki: [1.04737305 1.04637556 1.02531251 1.03980249 1.04611019 1.00557519 1.05281172 1.01086257 1.04820271 1.02630251]

Decision Tree Regressor:

	Błąd RMSE trening: 0.0
	Błąd RMSE test: 1.4958255008829713
	Wyniki kroswalidacji:

		Średnia: 1.4812743898817309
		Odchylenie standardowe: 0.04638256161096411
		Wyniki: [1.50887343 1.44477633 1.5509158  1.48903761 1.50242522 1.47601315 1.49523341 1.47477782 1.5045496  1.36614152]

Random Forest Regressor:

	Błąd RMSE trening: 0.46717510683821944
	Błąd RMSE test: 1.0891580129946803
	Wyniki kroswalidacji:

		Średnia: 1.1066827491292155
		Odchylenie standardowe: 0.022944494549647226
		Wyniki: [1.14262621 1.11033866 1.11365095 1.10415148 1.13758602 1.0747234 1.10233928 1.08156428 1.12428187 1.07556535]

RF GSCV Regressor:

	Błąd RMSE trening: 0.38927588342759734
	Błąd RMSE test: 1.0427901126489298
	Wyniki kroswalidacji:

		Średnia: 1.0484301826345184
		Odchylenie standardowe: 0.01701046097588534
		Wyniki: [1.0676943  1.04899351 1.05191613 1.05160395 1.0586865  1.02400685 1.05537385 1.02202774 1.07466483 1.02933415]

Lasso Regressor:

	Błąd RMSE trening: 1.156328659694595
	Błąd RMSE test: 1.1430864849765685
	Wyniki kroswalidacji:

		Średnia: 1.156255019501933
		Odchylenie standardowe: 0.017328587465414772
		Wyniki: [1.18718857 1.15645518 1.15094435 1.17835695 1.16493817 1.13283149 1.14973524 1.12737893 1.15373185 1.16098947]

LassoCV Regressor:

	Błąd RMSE trening: 1.0323533677567982
	Błąd RMSE test: 1.0153161619153646
	Wyniki kroswalidacji:

		Średnia: 1.034813444492904
		Odchylenie standardowe: 0.015987580541358157
		Wyniki: [1.04713074 1.0464945  1.0247456  1.04005519 1.04571174 1.00563546 1.05301873 1.01066614 1.04832968 1.02634667]

# Wnioski końcowe
## Klasyfikacja
Żaden z modeli nie osiągnął zadowalającego wyniku.
Zauważalne natomiast jest, że wyniki znacznie poprawiły się w drugim podejściu do brakujacych rekordów.

Prawdopodobnie dalsza analiza danych i ich przetworzenie pozwoliłyby na osiągnięcie lepszego wyniku.

##Regresja
Większość z modeli miała podobne wyniki, natomiast niezależnie od podejścia do brakująch rekordów, oraz przy użyciu PCA, najlepszym model okazywał się LassoCV.
Najprawdopodobniej wynika to z tego, że model ten ogranicza ilość predyktorów.

Dalsza redukcja cech przyniosłaby poprawę wyników końcowych.


time elapsed seconds:  1051
minutes =  17.516666666666666


