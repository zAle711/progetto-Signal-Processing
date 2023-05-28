import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from logistic_regression import sigmoid, cost_function, compute_hessian
from sklearn.preprocessing import StandardScaler
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

    # diagonal = np.dot(np.eye(numero_osservazioni), np.dot(s, (1-s)))  #(s * (1 - s))
    # H = np.dot(np.dot(X.T, diagonal), X)
    # return H

step = 0.001
for i in range(1, 3):
        hessian = compute_hessian(x_smaller, pesi_parametri)
        h = sigmoid(np.matmul(x_smaller,pesi_parametri))
        gradient = np.matmul(x_smaller.T, (h-y_smaller)) / n_osservazioni
        pesi_parametri = pesi_parametri - step * np.dot(np.linalg.inv(hessian), np.dot(x_smaller.T, (h - y_smaller)))
        print(cost_function(x_smaller, y_smaller, pesi_parametri))

# hessian = compute_hessian(x_smaller, pesi_parametri)
# h = sigmoid(np.matmul(x_smaller,pesi_parametri))
# print(h.shape, y_smaller.shape, x_smaller.T.shape, (h-y_smaller).shape)

# pesi_parametri = pesi_parametri - step * np.dot((np.linalg.inv(hessian), np.dot(x_smaller.T, (h - y_smaller)))
# print(cost_function(x_smaller, y_smaller, pesi_parametri))

# pd.DataFrame(hessian).info()
