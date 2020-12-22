import numpy as np
np.random.seed(0)

# Devuelve valores que suman uno.
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(sx):
    #Dado que estamos usando activación sigmoidea y necesitaremos
    #su gradiente mientras se propaga hacia atrás.
    #Encontrar el gradiente de la función sigmoidea y reducirlo a
    #una expresión simplificada.
    return sx * (1 - sx)

#funcion costo.
def costo(prediccion, real):
    return real - prediccion

entrada_xor = np.array([[0,0], [0,1], [1,0], [1,1]])
salida_xor = np.array([[0,1,1,0]]).T

#Usamos entrenamiento y prueba
X = entrada_xor
Y = salida_xor

#Forma del vector de peso
num_data, entrada_dim = X.shape

#Dimension de la capa oculta
capa_oculta = 5

#Peso entre las capas de entrada y la capa oculta.
W1 = np.random.random((entrada_dim, capa_oculta))

#Defina la forma del vector de salida
salida_dim = len(Y.T)

#Peso entre las capas ocultas y la capa de salida.
W2 = np.random.random((capa_oculta, salida_dim))

#Número de iteraciones
num_iteraciones = 10000

#Taza de aprendizaje
tasa_de_aprendizaje = 1.0

for iteracion in range(num_iteraciones):
    capa0 = X
    #Propagación hacia adelante
    
    #Dentro del perceptrón
    capa1 = sigmoid(np.dot(capa0, W1))
    capa2 = sigmoid(np.dot(capa1, W2))


    #Propagación hacia atrás (Y -> capa2)
    
    #Cuánto nos perdimos en las predicciones?
    capa2_error = costo(capa2, Y)

    #En qué dirección está el valor objetivo?
    #Estábamos realmente cerca?
    capa2_d = capa2_error * sigmoid_derivative(capa2)

    
    #Propagación hacia atrás (capa2 -> capa1)
    #Influencia de los pesos del valor de capa1 al error de capa2
    capa1_error = np.dot(capa2_d, W2.T)
    capa1_delta = capa1_error * sigmoid_derivative(capa1)
    
    #actualizamos pesos
    W2 +=  tasa_de_aprendizaje  * np.dot(capa1.T, capa2_d)
    W1 +=  tasa_de_aprendizaje  * np.dot(capa0.T, capa1_delta)


for x, y in zip(X, Y):
    #Introduzca la entrada predecida W1
    prediccion_capa1 = sigmoid(np.dot(W1.T, x))
    #Introduzca la entrada predecida W2
    prediccion = prediccion_capa2 = sigmoid(np.dot(W2.T, prediccion_capa1))
    #valores predecidos mayores al 50% comparados con y (valores de entrenamiento)
    print(int(prediccion > 0.5), y)