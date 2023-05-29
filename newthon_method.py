import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from logistic_regression import sigmoid, cost_function, compute_hessian
from sklearn.preprocessing import StandardScaler
from util import save_results_to_json
#Importo il dataset dal csv
data_training = pd.read_csv('test.csv')
x_tr = data_training.drop('diabetes', axis=1)
y_tr = data_training['diabetes']
#Normalizzo i dati
scaler = StandardScaler()
x_tr = scaler.fit_transform(x_tr)
#Aggiungo il bias
x_tr = np.hstack([np.ones([x_tr.shape[0],1]), x_tr])
#Seleziono una porzioni dei dati poichè il pc non è sufficientemente potente
n_osservazioni = 50000
x_smaller = x_tr[0:n_osservazioni, :]
y_smaller = y_tr[0:n_osservazioni]

pesi_parametri = np.zeros(9)
n_iterazioni = 1000;
n_parametri = 9

    # diagonal = np.dot(np.eye(numero_osservazioni), np.dot(s, (1-s)))  #(s * (1 - s))
    # H = np.dot(np.dot(X.T, diagonal), X)
    # return H

learning_rate = [0.01, 0.001, 0.0001]

for step in learning_rate:

    #inizializzo una matrice in cui salvo i valore del parametri ad ogni iterazione
    pesi_parametri = np.zeros([n_parametri, n_iterazioni])
    #inizializzo i primi parametri da dare all'algoritmo in maniere casuale
    pesi_parametri[:,0] = np.ones(n_parametri)
    #creo una lista in cui vado a salvare il valore della funzione costo ad ogni iterazione
    valore_funzione_costo = []
    #calcolo il valore della funzione costo con i primi parametri casuali
    valore_funzione_costo.append(cost_function(x_tr, y_tr, pesi_parametri[:,0]))

    for i in range(1, n_iterazioni-1):
        hessian = compute_hessian(x_smaller, pesi_parametri[:,i])
        h = sigmoid(np.matmul(x_smaller,pesi_parametri[:,i]))
        gradient = np.matmul(x_smaller.T, (h-y_smaller)) / n_osservazioni
        pesi_parametri[:,i+1] = pesi_parametri[:,i] - step * np.dot(np.linalg.inv(hessian), np.dot(x_smaller.T, (h - y_smaller)))
        costo = cost_function(x_smaller, y_smaller, pesi_parametri[:,i+1])
        print(f"Step-Size: {step} Costo: {costo}")
        valore_funzione_costo.append(costo)
    
    dictionary = {
                'step': step,
                'value_cost_function': valore_funzione_costo,
                'value_optimal_parameters': pesi_parametri[:, n_iterazioni-1].tolist()
        }
    save_results_to_json(dictionary, 'newthon_method')

# [Done] exited with code=0 in 53368.387 seconds