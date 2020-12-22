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

#NO UTILIZAMOS DATA REDUCTION



#DISCRETIZACION: agrupa datos para reducir su variabilidad porque los resultados de la clase son muy variados
from sklearn.preprocessing import  KBinsDiscretizer

#El número de contenedores a producir
#Método utilizado para codificar el resultado transformado: 

#Kmeans: la discretización se basa en los centroides de un procedimiento de agrupación de KMeans.
class_discr = KBinsDiscretizer(n_bins=5, encode='ordinal', strategy = "kmeans").fit_transform(data[['clase']])

#Tipo de Dato DataFrame
class_discr = pd.DataFrame(class_discr)
#Discretiza la columna MYCT para conseguir conjuntos de valores
class_discr = class_discr.rename(columns = {6: 'clase'})

#sobreescribimos la columna clase con los datos discretizados
data[['clase']] = class_discr

print(data)

"""
#NORMALIZACION
#La normalización es el proceso de escalar muestras individuales
#para tener una norma unitaria.
#Este proceso puede ser útil si planea utilizar una forma cuadrática
#como el producto punto o cualquier otro núcleo para cuantificar
#la similitud de cualquier par de muestras.

from sklearn import preprocessing
matriz_normal = preprocessing.normalize(modelo)

print('Normalizacion')
print(matriz_normal)
data.head(10)

#Fin Normalizacion
"""

#from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import export_text
from sklearn.model_selection import train_test_split

X=data[['MYCT', 'MMIN', 'MMAX', 'CACH', 'CHMIN','CHMAX']]
y=data['clase']

#80% Para entrenamiento y 20 % para prueba
#Dividimos aleatoriamente nuestros datos
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=0)

#max_depth: profundidad del árbol
#MODELO
#modelo = DecisionTreeRegressor(max_depth = 5, random_state = 0)

from sklearn import tree
modelo = tree.DecisionTreeClassifier(criterion='entropy', random_state=0)

# Aprendizaje
modelo.fit(X_train, y_train)

#print(f"Profundidad: {modelo.get_depth()}")

#Mostrar arbol
texto_modelo = export_text(decision_tree = modelo, feature_names = list(data.drop(columns = "clase").columns))
print(texto_modelo)







"""
#ARBOL DE DESICION
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

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
graph.render("CPU")"""
