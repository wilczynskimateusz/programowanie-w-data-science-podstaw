# -*- coding: utf-8 -*-
"""
Created on Thu Feb 13 23:19:44 2020

@author: blood
"""

# Biblioteki z których będziemy korzystać
from IPython.core.interactiveshell import InteractiveShell
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
# Jesli chcemy mieć kilka "outputów" z jednej komórki
InteractiveShell.ast_node_interactivity = "all"


# Ustawienie ogólnego stylu dla plotów z biblioteki seaborn
sns.set(style="whitegrid", color_codes=True)
fav = ["#DE357D", "#15B4D4", "#10659E", "#FFBE22", "#FF6A6C",
       "#69EAEC", "#95a5a6", "#34495e", "#EE152A", "#39AD00"]
sns.palplot(sns.color_palette(fav))
sns.set_context("talk", font_scale=1.2, rc={"grid.linewidth": 0.8})


# Implementacja pliku "titanic.csv" w zmiennej "titanic" jako DataFrame
titanic = pd.read_csv(
    "titanic.csv", sep='\t')
# Dodanie kolumny "Family Size" która jest suma "SibSp" i "Parch"
titanic['FamilySize'] = titanic.SibSp + titanic.Parch + 1

# Sprawdzenie struktury - liczba kolumn oraz liczba wierszy
titanic.shape

# Sprawdzenie, które dane są niekompletne
titanic.isnull().sum()

# Sprawdzamy jaki ubytek w zestawieniu do całości stanowi "Age"
round(30/(len(titanic["PassengerId"])), 4)*100
# Sprawdzamy jaki ubytek w zestawieniu do całości stanowi "Cabin"
round(125/len(titanic["PassengerId"]), 4)*100
# Sprawdzamy jaki ubytek w zestawieniu do całości stanowi "Embarked"
round(1/len(titanic["PassengerId"]), 4)*100


# Ubytek w "Age" stanowi ~20% więc warto to naprawić
# Obliczamy medianę dla kolumny 'Age'
titanic["Age"].median(skipna=True)

# Zastępujemy "Nan'y" w kolumnie "Age" wyliczoną medianą
titanic["Age"].fillna(26, inplace=True)

# Brakuje ~80% rekordów, co oznacza, że przypisywanie informacji i wykorzystywanie tej zmiennej
# jest nierozsądne. Usuniemy tą zmienną z naszego DataFrame.
titanic.drop('Cabin', axis=1, inplace=True)

# Sprawdzamy dominante dla kolumny "Embarked"
titanic.loc[:, "Embarked"].mode()

# Najwięcej osób wchodzi na pokład w Southhampton więc zastąpimy Nan'a wartością "S"
titanic["Embarked"].fillna("S", inplace=True)


# Ponowne sprawdzenie kompletności danych (po optymalizacji)
titanic.isnull().sum()

# Prezentacja naszego DataFrame
titanic.sample(5)

plt.figure(figsize=(25, 8), dpi=100)
titanic.Survived.value_counts().plot(kind='bar', color=fav[7], alpha=1)
plt.title("Przetrwanie, (0 = Zmarli / 1 = Ocaleni)")
plt.gca().set(ylabel='Liczba')
sns.despine()

plt.figure(figsize=(25, 8), dpi=100)
titanic.Pclass.value_counts().plot(kind='bar', color=fav[-4], alpha=1)
plt.title("Klasa (Pclass)")
plt.gca().set(ylabel='Liczba')
sns.despine()

%matplotlib inline
plt.figure(figsize=(25, 10), dpi=100)
sns.kdeplot(titanic.Age, shade=True, alpha=0.7, color=fav[0], label='Wiek')
plt.axvline(titanic.Age.median(), color=fav[-1], label='Mediana', ls='dashed')
plt.axvline(titanic.Age.mean(), color=fav[-2], label='Średnia', ls='dashed')
plt.legend()
plt.title("Wiek, (Age)")
plt.gca().set(ylabel='Odsetek')
sns.despine()

plt.figure(figsize=(25, 8), dpi=100)
titanic.Sex.value_counts().plot(kind='bar', color=fav[4], alpha=1)
plt.title("Płeć, (Sex)")
plt.gca().set(ylabel='Liczba')
sns.despine()


plt.figure(figsize=(25, 8), dpi=100)
titanic.Embarked.value_counts().plot(kind='bar', color=fav[3], alpha=1)
plt.title("Miejsce wypłynięcia, (Embarked)")
plt.gca().set(ylabel='Liczba')
sns.despine()

plt.figure(figsize=(25, 10), dpi=100)
sns.kdeplot(titanic.Fare[titanic.Pclass == 1].apply(
    lambda x: 80 if x > 80 else x), shade=True, alpha=0.7, color=fav[0], label='1st Class')
sns.kdeplot(titanic.Fare[titanic.Pclass == 2].apply(
    lambda x: 80 if x > 80 else x), shade=True, alpha=0.7, color=fav[4], label='2nd Class')
sns.kdeplot(titanic.Fare[titanic.Pclass == 3].apply(
    lambda x: 80 if x > 80 else x), shade=True, alpha=0.7, color=fav[5], label='3rd Class')
plt.axvline(titanic.Fare.median(), color=fav[-1], label='Mediana', ls='dashed')
plt.axvline(titanic.Fare.mean(), color=fav[-2], label='Średnia', ls='dashed')
plt.legend()
plt.title("Rozkład cen biletów ze względu na wykupiona klasę podróży, (Fare)")
plt.gca().set(xlabel='Cena', ylabel='Odsetek')
sns.despine()

plt.figure(figsize=(25, 10), dpi=100)
sns.kdeplot(titanic.Age[titanic.Sex == 'male'], shade=True,
            color=fav[2], alpha=0.7, label='Mężczyzna')
sns.kdeplot(titanic.Age[titanic.Sex == 'female'],
            shade=True, color=fav[-2], alpha=0.7, label='Kobieta')
plt.legend()
plt.title("Rozkład wieku według płci")
plt.gca().set(xlabel='Wiek', ylabel='Odsetek')
sns.despine()

plt.figure(figsize=(25, 8), dpi=100)
titanic.SibSp.value_counts().plot(kind='bar', alpha=1, color=fav[6])
titanic.Parch.value_counts().plot(kind='bar', alpha=0.6, color=fav[1])
plt.legend()
plt.title("Liczba rodzeństwa, małżonków, rodziców i dzieci")
plt.gca().set(ylabel='Liczba')
sns.despine()


titanic['Survived'].value_counts()

from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "none"
# 1 klasa
plt.figure(figsize=(25, 8), dpi=100)
titanic.Survived[titanic.Pclass == 1].value_counts().sort_index().plot(
    kind='bar', alpha=0.85, color=fav[0], label='1st Class')
plt.title("Stopien przezycia w 1 klasie")
plt.gca().set(ylabel='Liczba')
sns.despine()

# 2 klasa
plt.figure(figsize=(25, 8), dpi=100)
titanic.Survived[titanic.Pclass == 2].value_counts().sort_index().plot(
    kind='bar', alpha=0.85, color=fav[1], label='2nd Class')
plt.title("Stopien przezycia w 2 klasie")
plt.gca().set(ylabel='Liczba')
sns.despine()

# 3 klasa
plt.figure(figsize=(25, 8), dpi=100)
titanic.Survived[titanic.Pclass == 3].value_counts().sort_index().plot(
    kind='bar', alpha=0.85, color=fav[4], label='3rd Class')
plt.title("Stopien przezycia w 3 klasie")
plt.gca().set(ylabel='Liczba')
sns.despine()


plt.figure(figsize=(25, 8), dpi=100)
plt.bar(np.array([0, 1])-0.25, titanic.Survived[titanic.Sex == 'male'].value_counts(
).sort_index(), width=0.25, color=fav[2], label='Mężczyzna', alpha=0.7)
plt.bar(np.array([0, 1]), titanic.Survived[titanic.Sex == 'female'].value_counts(
).sort_index(), width=0.25, color=fav[5], label='Kobieta', alpha=0.7)
plt.xticks(np.arange(0, 2, 1))
plt.legend()
plt.title("Stopien przeżycia ze względu na płeć (Sex)")
plt.gca().set(ylabel='Liczba')
sns.despine()

plt.figure(figsize=(25, 10), dpi=100)
sns.kdeplot(titanic.Age[titanic.Survived == 0],
            shade=True, color=fav[-2], label='Zmarli')
sns.kdeplot(titanic.Age[titanic.Survived == 1],
            shade=True, color=fav[-1], label='Ocaleni')
plt.legend()
plt.title("Stopien przeżycia w zestawieniu z wiekiem")
plt.gca().set(ylabel='Odsetek', xlabel='Wiek')
sns.despine()

# Cherbourg
plt.figure(figsize=(25, 8), dpi=100)
plt.hist(titanic.Survived[pd.Categorical(
    titanic.Embarked).codes == 1], color=fav[1],)
plt.title("Cherbourg (C) / (0-zmarli/1-ocaleni)")
plt.gca().set(ylabel='Liczba')
sns.despine()

# Queenstown
plt.figure(figsize=(25, 8), dpi=100)
plt.hist(titanic.Survived[pd.Categorical(
    titanic.Embarked).codes == 1], color=fav[4],)
plt.title("Queenstown (Q) / (0-zmarli/1-ocaleni)")
plt.gca().set(ylabel='Liczba')
sns.despine()

# Southampton
plt.figure(figsize=(25, 8), dpi=100)
plt.hist(titanic.Survived[pd.Categorical(
    titanic.Embarked).codes == 2], color=fav[5],)
plt.title("Southampton (S) / (0-zmarli/1-ocaleni)")
plt.gca().set(ylabel='Liczba')
sns.despine()

plt.figure(figsize=(25, 10), dpi=100)
sns.kdeplot(titanic.FamilySize[titanic.Survived == 0],
            shade=True, color=fav[-2], label='Zmarli')
sns.kdeplot(titanic.FamilySize[titanic.Survived == 1],
            shade=True, color=fav[-1], label='Ocaleni')
plt.title('Przeżywalność ze względu na wielkość rodziny')
plt.gca().set(ylabel='Odsetek')
sns.despine()
plt.legend()
plt.show()

plt.figure(figsize=(25, 8), dpi=100)
titanic.FamilySize.value_counts().plot(kind='bar', alpha=1, color=fav[5])
plt.legend()
plt.title("Liczebność rodziny")
plt.gca().set(xlabel='Ilość osób', ylabel='Liczba')
sns.despine()


"""
Wnioski:
Na podstawie powyższych danych, możemy stwierdzić 3 główne czynniki wpływające na przeżywalność osób na pokładzie:
umiejscowienie na pokładzie - osoby znajdujące się w pierwszej klasie, miały największe szanse na przeżycie, wiązało się to z konstrukcją i umiejscowieniem kabin 1 klasy na statku najbliżej kajut. Ponadto, w obliczu ograniczonej ilości kajut ratunkowych, pierwsze z nich były ładowane częściowo, gdy osoby z trzeciej klasy uzyskały już dostęp do kajut, było ich zbyt mało. Najmniejsza przeżywalność w trzeciej klasie wynikła z przestarzałych przepisów oddzielania pasażerów trzeciej klasy od pozostałych.
płeć - kobiety były  traktowane były priorytetowo przy ewakuacji, zwłaszcza w początkowym okresie ewakuacji 1 i 2 klasy.
wiek - dzieci, podobnie jak kobiety, traktowane były priorytetowo przy ewakuacji, zwłaszcza w początkowym okresie ewakuacji 1 i 2 klasy.
"""