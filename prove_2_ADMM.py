import numpy as np
import pandas as pd
from scipy.optimize import minimize
from multiprocessing import Pool
from sklearn.preprocessing import StandardScaler
from logistic_regression import cost_function, sigmoid

n_iterazioni = 1000
n_agenti = 4
n_parametri = 9

# u = np.zeros([n_parametri, n_iterazioni, n_agenti])

def update_local_variable(x_tr, y_tr, pesi_parametri, z, u, rho):
    """_summary_
    Args:
        pesi_parametri : n_parametri X n_agenti
        z : n_parametri
        u : n_parametri X n_agenti
    """

    dati_per_agente = int(x_tr.shape[0]/n_agenti)
    
    for i in range(0, n_agenti):
        #print(f"Agente: {i}, range dati assegnato all'agente: {(dati_per_agente*i,dati_per_agente*(i+1))}")
        x_i = x_tr[dati_per_agente*i:dati_per_agente*(i+1), :]
        y_i = y_tr[dati_per_agente*i:dati_per_agente*(i+1)]

        n_osservazioni = x_i.shape[0]

        pesi_parametri_gradiente = np.zeros([n_parametri, n_iterazioni])
        pesi_parametri_gradiente[:, 0] = pesi_parametri[:,i]
        # print(pesi_parametri_gradiente.shape)
        valore_funzione_costo = []
        valore_funzione_costo.append(cost_function(x_tr, y_tr, pesi_parametri[:, i]))

        # gradient = np.dot(x_tr.T, (h-y_tr)) / n_osservazioni
        #print(pesi_parametri[:, 0].shape)
        for j in range(0, n_iterazioni-1):

            if False:
                print(f"Iterazione n: {j}")
                print("s: ", s.shape)
                print("G: ",np.dot(x_i.T, (s - y_i)).shape)
                print("w: ", pesi_parametri[:,0].shape)
                print("x_tr.T: ", x_i.T.shape, " y_tr: ", y_i.shape)

            s = sigmoid(np.dot(x_i,pesi_parametri[:,i]))
            #sicuramente z è quello dell'iterazione generale
            pesi_parametri_gradiente[:, j+1] = pesi_parametri_gradiente[:, j]  - (0.01) * (1/x_i.shape[0]) * (np.dot(x_i.T, (s - y_i))) #+ rho *  (pesi_parametri_gradiente[:, j] - z - u[:,j] )
        pesi_parametri[:, i] = pesi_parametri_gradiente[:,-1]
    return pesi_parametri

# Step-Size: 0.01 Costo: 0.5295995155297352
# s:  (60390,)
# x_tr.T:  (9, 60390)  y_tr:  (60390,)
# G:  (9,)
# w:  (9,)

def update_g(pesi, u):
    """
    Args:
        pesi : n_parametri X n_agenti
        u : n_parametri X n_agenti
    """
    # print(pesi[:,1].shape)
    z = np.zeros([n_parametri,1])
    #Per calcolarla ho bisogno della media dei pesi sommata alla media di u
    media_pesi = np.zeros(n_parametri)
    media_u = np.zeros(n_parametri)

    for i in range(0, n_agenti):
        media_pesi += pesi[:,i] 
        media_u += u[:,i]
    
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
        new_u[:,i] = u[:,i] + pesi_parametri[:,i] - z
    
    return new_u

def ADMM(x_tr, y_tr):
    #Iniziallizzo le matrici dei pesi, la variaible globale z e la variabile u
    pesi_parametri = np.zeros([n_parametri, n_iterazioni, n_agenti])
    z = np.zeros([n_parametri, n_iterazioni])
    u = np.zeros([n_parametri, n_iterazioni, n_agenti])
    #Inizializzo casualmente i primi pesi e valori di u
    pesi_parametri[:,0,:] = np.random.randn(n_parametri,n_agenti)
    u[:,0,:] = np.random.randn(n_parametri,n_agenti)
    #Aggiorno la variaible globale con i nuovi valori
    z[:,0] = update_g(pesi_parametri[:,0,:], u[:,0,:])
    #Inizializzo matrice per raccogliere i valori della funzione costo
    costo_logistic = np.zeros([n_agenti, n_iterazioni])
    costo_logistic[:,0] = cost_function(x_tr, y_tr, pesi_parametri[:,0,:])

    rho = 10

    for i in range(0,10):
        pesi_parametri[:,i+1,:] = update_local_variable(x_tr, y_tr, pesi_parametri[:,i,:], z[:,i], u[:,i,:], rho )
        z[:, i+1] = update_g(pesi_parametri[:,i+1,:], u[:,i,:])
        u[:,i+1,:] = update_u(u[:,i,:],pesi_parametri[:,i+1,:], z[:,i+1])
        costo_logistic[:,i+1] = cost_function(x_tr, y_tr, pesi_parametri[:,i+1,:])
    
    print(pesi_parametri[:, 9, 0])
        


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
    #update_local_variable(x_tr, y_tr)

    ADMM(x_tr, y_tr)
