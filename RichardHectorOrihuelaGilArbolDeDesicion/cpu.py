#Librerias
import pandas as pd

#<LECTURA>
#importar data-set
data = pd.read_csv("cpu.csv")

#media, varianza, min, max, ...
print(data.describe())

#PREPROCESSING
#contar valores perdidos en columna
print("*****Datos Faltantes*****")
missing_values_count = data.isnull().sum()
print(missing_values_count)
#no faltan valores

#NO ES NECESARIO IMPUTACION, no hay datos perdidos

#<HISTOGRAMAS>
#from pylab import *

#MYCT
#nums = data["MYCT"]

#MMIN
#nums = data["MMIN"]

#MMAX
#nums = data["MMAX"]

#CACH
#nums = data["CACH"]

#CHMIN
#nums = data["CHMIN"]

#CHMAX
#nums = data["CHMAX"]
#hist(nums)
#show()
#</HISTOGRAMAS>

#No es necesario NORMALIZAR, no necesitamos agrupar en rango de datos

#No DISCRETIZAMOS PORQUE NO SON DATOS CONTINUOS

#NO UTILIZAMOS DATA REDUCTION

X=data.iloc[:,:-1]
y=data.clase

print(X.head)
print(y.head)



#ARBOL DE DESICION
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt # visualisacion de datos
from sklearn import tree

#Usa el algoritmo C4.5

#>>>Arbol de decisión<<<
from sklearn import tree
arbol = tree.DecisionTreeClassifier(random_state=0)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#aprendizaje
arbolAjuste = arbol.fit(X_train, y_train)

#predecir
print(">>>Arbol de decision")
y_predecido_arbol = arbolAjuste.predict(X_test)
#print(arbol.predict([[50.,1.,168.,0,38.,1.,276000.,1.1,137.,1.,0]]))
#tree.plot_tree(arbol)

#score
print(accuracy_score(y_test, y_predecido_arbol))

#Grafica del arbol de desicion
import graphviz 
dot_data = tree.export_graphviz(arbol, out_file=None) 
graph = graphviz.Source(dot_data)
graph.render("CPU")