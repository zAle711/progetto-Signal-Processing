import numpy as np
import pandas as pd
from scipy.optimize import minimize
from multiprocessing import Pool
from sklearn.preprocessing import StandardScaler
from logistic_regression import cost_function, sigmoid

n_agenti = 4
n_iterazioni = 100
n_parametri = 9

u = np.zeros([n_parametri, n_iterazioni, n_agenti])
z = np.zeros([n_parametri, n_iterazioni])

def update_local_variable(x_tr, y_tr):
    pippo = 1
    soluzioni = np.zeros([x_tr.shape[1], n_agenti])
    dati_per_agente = int(x_tr.shape[0]/n_agenti)
    for i in range(0, n_agenti):
        print(f"Agente: {i}, range dati assegnato all'agente: {(dati_per_agente*i,dati_per_agente*(i+1))}")
        x_i = x_tr[dati_per_agente*i:dati_per_agente*(i+1), :]
        y_i = y_tr[dati_per_agente*i:dati_per_agente*(i+1)]

        n_osservazioni = x_i.shape[0]

        rho = 1
        u = np.zeros([n_parametri, n_iterazioni])
        u[:,0] = np.random.randn(n_parametri)
        pesi_parametri = np.zeros([n_parametri, n_iterazioni])
        pesi_parametri[:,0] = np.ones(n_parametri)
        valore_funzione_costo = []
        valore_funzione_costo.append(cost_function(x_tr, y_tr, pesi_parametri[:,0]))  

        # gradient = np.dot(x_tr.T, (h-y_tr)) / n_osservazioni
        print(pesi_parametri[:, 0].shape)
        for j in range(0, n_iterazioni-1):

            if False:
                print(f"Iterazione n: {j}")
                print("s: ", s.shape)
                print("G: ",np.dot(x_i.T, (s - y_i)).shape)
                print("w: ", pesi_parametri[:,0].shape)
                print("x_tr.T: ", x_i.T.shape, " y_tr: ", y_i.shape)

            s = sigmoid(np.dot(x_i,pesi_parametri[:,0]))
            #sicuramente z è quello dell'iterazione generale
            pesi_parametri[:, j+1] = pesi_parametri[:, j] - (0.01) * (1/n_osservazioni) * (np.dot(x_i.T, (s - y_i))) + rho *  (pesi_parametri[:, j] - z[:,j] - u[:,j] )
# Step-Size: 0.01 Costo: 0.5295995155297352
# s:  (60390,)
# x_tr.T:  (9, 60390)  y_tr:  (60390,)
# G:  (9,)
# w:  (9,)




if __name__ == '__main__': 
    #Importo il dataset dal csv
    data_training = pd.read_csv('test.csv')
    x_tr = data_training.drop('diabetes', axis=1)
    y_tr = data_training['diabetes'].to_numpy()
    #Normalizzo i dati
    scaler = StandardScaler()
    x_tr = scaler.fit_transform(x_tr)

    # Aggiungo il bias
    x_tr = np.hstack([np.ones([x_tr.shape[0],1]), x_tr])
    # Chiamata alla funzione ADMM per addestrare il modello logistic regression
    update_local_variable(x_tr, y_tr)
