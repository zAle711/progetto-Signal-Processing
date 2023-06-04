import numpy as np
import pandas as pd
from scipy.optimize import minimize
from multiprocessing import Pool
from sklearn.preprocessing import StandardScaler
from logistic_regression import cost_function, sigmoid
from util import save_results_to_json

n_iterazioni = 500

n_agenti = 4
n_parametri = 9


def update_local_variable(x_tr, y_tr, pesi_parametri, z, u, rho):
    """_summary_
    Args:
        pesi_parametri : n_parametri X n_agenti
        z : n_parametri
        u : n_parametri X n_agenti
    """

    dati_per_agente = int(x_tr.shape[0]/n_agenti)
    n_iterazioni = 500
    for i in range(0, n_agenti):
        #print(f"Agente: {i}, range dati assegnato all'agente: {(dati_per_agente*i,dati_per_agente*(i+1))}")
        x_i = x_tr[dati_per_agente*i:dati_per_agente*(i+1), :]
        y_i = y_tr[dati_per_agente*i:dati_per_agente*(i+1)]

        n_osservazioni = x_i.shape[0]

        pesi_parametri_gradiente = np.zeros([n_parametri, n_iterazioni])
        pesi_parametri_gradiente[:, 0] = pesi_parametri[:, i]
        valore_funzione_costo = []
        valore_funzione_costo.append(
            cost_function(x_tr, y_tr, pesi_parametri[:, i]))

        for j in range(0, n_iterazioni-1):

            if False:
                print(f"Iterazione n: {j}")
                print("s: ", s.shape)
                print("G: ", np.dot(x_i.T, (s - y_i)).shape)
                print("w: ", pesi_parametri[:, 0].shape)
                print("x_tr.T: ", x_i.T.shape, " y_tr: ", y_i.shape)

            s = sigmoid(np.dot(x_i, pesi_parametri[:, i]))

            pesi_parametri_gradiente[:, j+1] = pesi_parametri_gradiente[:, j] - (0.001) * (1/n_osservazioni) * (np.dot(x_i.T, (s - y_i))) + rho * (pesi_parametri_gradiente[:, j] - z - u[:, i])

        pesi_parametri[:, i] = pesi_parametri_gradiente[:, -1]
    return pesi_parametri

def update_z(pesi, u):
    """
    Args:
        pesi : n_parametri X n_agenti
        u : n_parametri X n_agenti
    """
    # Per calcolarla ho bisogno della media dei pesi sommata alla media di u
    media_pesi = np.zeros(n_parametri)
    media_u = np.zeros(n_parametri)

    for i in range(0, n_agenti):
        media_pesi += pesi[:, i]
        media_u += u[:, i]

    return (media_pesi + media_u) / n_agenti


def update_u(u, pesi_parametri, z):
    """
    Args:
        u: n_parametri x n_agenti
        pesi_parametri : n_parametri x n_agenti
        z : n_parametri
    """

    new_u = np.zeros([n_parametri, n_agenti])

    for i in range(n_agenti):
        new_u[:, i] = u[:, i] + pesi_parametri[:, i] - z

    return new_u


def ADMM(x_tr, y_tr):
    # Iniziallizzo le matrici dei pesi, la variaible globale z e la variabile u
    pesi_parametri = np.zeros([n_parametri, n_iterazioni, n_agenti])
    z = np.zeros([n_parametri, n_iterazioni])
    u = np.zeros([n_parametri, n_iterazioni, n_agenti]) - 1
    # Inizializzo casualmente i primi pesi e valori di u
    pesi_parametri[:, 0, :] = np.random.randn(n_parametri, n_agenti)
    u[:, 0, :] = np.random.randn(n_parametri, n_agenti)
    # Aggiorno la variaible globale con i nuovi valori
    z[:, 0] = update_z(pesi_parametri[:, 0, :], u[:, 0, :])

    # print(costo_logistic[:,0])

    rho_list = [0, 0.00001]
    for rho in rho_list:

        # Inizializzo matrice per raccogliere i valori della funzione costo
        costo_logistic = np.zeros([n_agenti, n_iterazioni])
        costo_logistic[:, 0] = cost_function(x_tr, y_tr, pesi_parametri[:, 0, :], n_agenti)

        for i in range(0, n_iterazioni-1):
            print(f"Iterazione n. {i}")
            pesi_parametri[:, i+1, :] = update_local_variable(
                x_tr, y_tr, pesi_parametri[:, i, :], z[:, i], u[:, i, :], rho)
            z[:, i+1] = update_z(pesi_parametri[:, i+1, :], u[:, i, :])
            u[:, i+1, :] = update_u(u[:, i, :], pesi_parametri[:, i+1, :], z[:, i+1])
            costo_logistic[:, i+1] = cost_function(x_tr, y_tr, pesi_parametri[:, i+1, :], n_agenti)

        
        costi = []
        for j in range(0, n_iterazioni):
            costi.append(costo_logistic[:, j].tolist())

        #Salvo i risultati ottenuti in un file json
        dictionary = {
            'rho': rho,
            'value_cost_function': costi,
            'value_optimal_parameters': [pesi_parametri[:, n_iterazioni-1, j].tolist() for j in range(0, n_agenti)],
            'z': z[:, -1].tolist()
        }
        save_results_to_json(dictionary, 'ADMM')


if __name__ == '__main__':
    # Importo il dataset dal csv
    data_training = pd.read_csv('test.csv')
    x_tr = data_training.drop('diabetes', axis=1)
    y_tr = data_training['diabetes'].to_numpy()
    # Normalizzo i dati
    scaler = StandardScaler()
    x_tr = scaler.fit_transform(x_tr)

    # Aggiungo il bias
    x_tr = np.hstack([np.ones([x_tr.shape[0], 1]), x_tr])
    # Chiamata alla funzione ADMM per addestrare il modello logistic regression
    ADMM(x_tr, y_tr)
