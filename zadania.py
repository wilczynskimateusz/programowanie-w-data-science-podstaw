# -*- coding: utf-8 -*-
"""
Created on Tue Feb 11 20:41:02 2020

@author: mattw
"""
from collections import Counter
import math
import numpy as np
"""
Zadanie 2
Korzystając z pojęcia funkcji utwórz skrypt, który będzie miał możliwość
zamiany temperatury pomiędzy skalami Celsjusza i Fahrenheita (w obie strony). 
C = (F-32)x(5/9), F = (C*9/5)+32
"""
def convertTemperature(temperature, direction):
    if direction == 'c':
        output = ((float(temperature) - 32)*(5/9))
    elif direction == 'f':
        output = (float(temperature) * (5/9))
    return round(output, 2)
    
direction = input('Podaj kierunek konwersji celsjusz (c), lub farenheit (f):\n')
temperature = input('Podaj wartosc temepratury do skonwertowania:\n')

print(convertTemperature(temperature,direction))

"""
Zadanie 4
Utwórz skrypt do znajdowania miejsc zerowych trójmianu kwadratowego 
x1 = (-b+sqrt(b*b-4*a*c))/(2*a)
x2 = (-b-sqrt(b*b-4*a*c))/(2*a)
"""
def zeroOfFunction(a,b,c):
    delta = b**2 - 4*a*c
    if delta > 0:
        return (-b-delta**0.5)/2*a, (-b+delta**0.5)/2*a
    elif delta == 0:
        return -b/2*a
    else:
        return None
    
print(zeroOfFunction(1,2,1))

"""
Zadanie 8
W klasie przeprowadzono sprawdzian, za który uczniowie mogli otrzymać 
maksymalnie 40 punktów. Skala ocen w szkole jest następująca: 0-39% - ndst, 
40-49% - dop, 50-69% - dst, 70-84% - db, 85-99% - bdb, 100% - cel. Utwórz 
skrypt z interfejsem tekstowym, który na podstawie podanej liczby punktów ze 
sprawdzianu wyświetli ocenę jaka się należy (skorzystaj z konstrukcji if, 
elif, else)
"""
score = input('Podaj liczbe punktow:\n')
percent = int(float(score)*100/40)
print('Zdobyles '+ str(percent) + '% punktow')
if percent > 100:
    print('Nieprawidlowa ilosc punktow')
elif percent < 0:
    print('Nieprawidlowa ilosc punktow')
elif percent < 40:
    print('ndst')
elif percent < 50:
    print('dop')
elif percent < 70:
    print('dst')
elif percent < 85:
    print('db')
elif percent < 100:
    print('bdb')
else:
    print('cel')
    
    
"""
Zadanie 11
Utwórz skrypt z interfejsem tekstowym który obliczy silnię od danego 
argumentu. Wykonać zadanie na dwa sposoby - iteracyjnie i rekurencyjnie
"""



def recursive_factorial(n):
  if n == 0:
    return 1
  else:
    return n *recursive_factorial(n-1)

def iterative_factorial(n):  
    if n<2: 
        return 1
    else:
        for i in range(2,n):
            n*=i
        return n

value = int(input("Podaj liczbe\n"))
print("Silnia z uzyciem math: " + str(math.factorial(value)))
print("Silnia z uzyciem wlasnej funkcji iteracyjnej: " + str(iterative_factorial(value)))
print("Silnia z uzyciem wlasnej funkcji rekurencyjnej: " + str(recursive_factorial(value)))

    
"""
Zadanie 14
Utworzyć skrypt z interfejsem tekstowym, który będzie zwracać wiersz n-tego 
rzędu z trójkąta Pascala (użytkownik podaje n, program zwraca odpowiadający 
wiersz trójkąta)
"""

def triangle(n):
    output = []
    current = [1]
    for i in range(0, n):
        output.append(current)
        current = newrow(current)
    return output
def newrow(row):
    output = []
    previous = 0
    for i in row:
        output.append(i + previous)
        previous = i
    output.append(previous)
    return output


value = int(input('Podaj liczbę: '))
print('Wynik to: ')
print(triangle(value))

"""
Zadanie 17
Utwórz funkcję, która będzie generować listy danych do wykreślenia w oparciu o:
a) fukcję liniową ax+b
b) funkcję kwadratową ax^2+bx+c
c) funkcję odwrotnie-potęgową a/x^n
Każda z fukcji powinna przyjmować parametry równania, natomiast zwracać 
powinna dwie listy - x i y, które następnie będzie można wykreślić na wykresie

"""


def linearFunciton(a,b):
    x = np.arange(-20,20,0.1)
    return x, list(map(lambda y: a*y+b, x))

def squareFunction(a,b,c):
    x = np.arange(-20,20,0.1)
    return x, list(map(lambda y: a*y**2+b*y+c, x))


def inverseExponentialFunction(a,n):
    x = list(np.arange(-2,2,0.5))
    if 0 in x: x.remove(0)
    return x, list(map(lambda y: a/(y**n), x))

inverseExponentialFunction(1,1)

"""
Zadanie 19
Korzystając ze słownika, utwórz funkcję, która będzie zwracać liczbę dni 
danego miesiąca w roku
"""

def monthlenght(month):
    months = {
        'styczen': '31',
        'luty': '29',
        'marzec': '31',
        'kwiecien': '30',
        'maj': '31',
        'czerwiec': '30',
        'lipiec': '31',
        'sierpien': '31',
        'wrzesien': '30',
        'pazdziernik': '31',
        'listopad': '30',
        'grudzien': '31'
    }

    month = month.strip().lower()
    return (months[month])

month = input('Podaj miesiac:\n')
print(monthlenght(month))


"""
Zadanie 23
Utwórz funkcję, która jako argument będzie przyjmować listę liczb 
zmiennoprzecinkowych, a jej wynikiem będzie dominanta (moda). 
Skorzystaj z obiektu Counter i jego metody most_common z pakietu collections
"""

def dominant(numbers):
    return Counter(numbers).most_common(1)[0][0]

print(dominant([1, 3, 2, 4, 3, 4, 4, 5, 4, 7, 2, 1, 6, 8]))


"""
Zadanie 24
Utwórz fukcję, która jako argument będzie przyjmować listę liczb 
zmiennoprzecinkowych, a jej wynikiem będzie odchylenie standardowe średniej
"""

def mean(data):
    return float(sum(data) / len(data))

def variance(data):
    mu = mean(data)
    return mean([(x - mu) ** 2 for x in data])

def stddev(data):
    return math.sqrt(variance(data))

print(stddev([1, 3, 2, 4, 3, 4, 4, 5, 4, 7, 2, 1, 6, 8]))

"""
Zadanie 25
Utwórz fukcję, która jako argument będzie przyjmować listę liczb 
zmiennoprzecinkowych, a jej wynikiem będzie drugi moment centralny (wariancja).
"""

def skewness(listOfNumbers):
    average = sum(listOfNumbers)/len(listOfNumbers)
    return round(sum(map(lambda x: (x-average)**2 ,listOfNumbers))/len(listOfNumbers),3)
    
skewness([12, 1.32, 1, 1.12])

"""
Zadanie 26
Utwórz funkcję, która jako argument będzie przyjmować listę liczb 
zmiennoprzecinkowych, a jej wynikiem będzie trzeci moment centralny (skośność)
"""
def skewness2(listOfNumbers):
    average = sum(listOfNumbers)/len(listOfNumbers)
    return round(sum(map(lambda x: (x-average)**3 ,listOfNumbers))/len(listOfNumbers),3)

    
skewness2([12, 1.32, 1, 1.12])

"""
Zadanie 28
Utwórz funkcję, która jako argument będzie przyjmować dwie listy o równej 
liczbie elementów, a jej wynikiem będzie współczynnik korelacji
"""
def corelation(x, y):
    if len(x) != len(y):
        return('Listy różnej długoci')
    else:
        sumaXY = 0
        sumaX = 0
        sumaY = 0
        sumaX2 = 0
        sumaY2 = 0

        for i in range(len(x)):
            sumaXY += x[i]*y[i]
            sumaX += x[i]
            sumaY += y[i]
            sumaX2 += x[i]**2
            sumaY2 += y[i]**2

        c1 = (len(x) * sumaXY - sumaX*sumaY)
        c2 = math.sqrt((len(x) * sumaX2 - sumaX**2)*(len(y) * sumaY2 - sumaY**2))

        return c1/c2
corelation([1,1,5,99,101],[65,33,4,5,6])

"""
Zadanie 34
Utwórz klasę Kwadrat z konstruktorem ustalającym jego bok oraz metodami: 
wyświetlającymi wartość tego boku, obliczającymi pole i obwód figury
"""

class kwadrat():
    def __init__(self, side = 10):
        self.side = side
        
    def show (self):
        return "Kwadrat o długości sideu {}".format(self.side)
        
    def perimeter  (self):
        return 4*self.side

    def field  (self):
        return self.side**2
    
kw = kwadrat()      
print(kw.show())     
print(kw.perimeter ())
print(kw.field ())