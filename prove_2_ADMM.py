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

# u = np.zeros([n_parametri, n_iterazioni, n_agenti])

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
            #print(((pesi_parametri_gradiente[:, j] - z - u[:,i] ) * rho).shape)
            # print(pesi_parametri_gradiente[:, j] , z - u[:,j])
            pesi_parametri_gradiente[:, j+1] = pesi_parametri_gradiente[:, j]  - (0.01) * (1/n_osservazioni) * (np.dot(x_i.T, (s - y_i))) + rho *  (pesi_parametri_gradiente[:, j] - z - u[:,i] )
    
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
    u = np.zeros([n_parametri, n_iterazioni, n_agenti]) -1
    #Inizializzo casualmente i primi pesi e valori di u
    pesi_parametri[:,0,:] = np.random.randn(n_parametri,n_agenti)
    u[:,0,:] = np.random.randn(n_parametri,n_agenti)
    #Aggiorno la variaible globale con i nuovi valori
    z[:,0] = update_g(pesi_parametri[:,0,:], u[:,0,:])

    #print(costo_logistic[:,0])

    rho_list = [0]
    for rho in rho_list:

        #Inizializzo matrice per raccogliere i valori della funzione costo
        costo_logistic = np.zeros([n_agenti, n_iterazioni])
        costo_logistic[:,0] = cost_function(x_tr, y_tr, pesi_parametri[:,0,:],n_agenti)

        for i in range(0,n_iterazioni-1):
            print(f"Iterazione n. {i}")
            pesi_parametri[:,i+1,:] = update_local_variable(x_tr, y_tr, pesi_parametri[:,i,:], z[:,i], u[:,i,:], rho)
            z[:, i+1] = update_g(pesi_parametri[:,i+1,:], u[:,i,:])
            u[:,i+1,:] = update_u(u[:,i,:],pesi_parametri[:,i+1,:], z[:,i+1])
            costo_logistic[:,i+1] = cost_function(x_tr, y_tr, pesi_parametri[:,i+1,:],n_agenti)


        for i in range(0, len(rho_list)):
            costi = []
            for j in range(0, n_iterazioni):
                costi.append(costo_logistic[:, j].tolist())

            dictionary = {
                        'rho': i,
                        'value_cost_function': costi, 
                        'value_optimal_parameters': pesi_parametri[:, n_iterazioni-1, i].tolist(),
                        'z': z[:,-1].tolist()
                }
    save_results_to_json(dictionary, 'ADMM')
    


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
    # import seaborn as sns
    # import matplotlib.pyplot as plt
    
    # df = pd.read_csv('test.csv')
    # y_test = df['diabetes']
    # x_test = df.drop('diabetes', axis=1)
    # x_test = np.hstack([np.ones([x_test.shape[0],1]), x_test])
    
    # w = np.array([0.028360005854234732, 1.259239505968353, -0.07212441849704278, -0.06978846415216239, 0.7399855669257183, 3.08033507866354, 1.7714468607855773, -0.2213661744046548, -0.08539134234828893])
    # y_predicted = sigmoid(np.dot(x_test,w)) > 0.5
    # # print(f"x_tr {x_tr.shape} pesi_parametri {pesi_parametri[:,n_iterazioni-1].shape}")
    # y_predicted = y_predicted.astype(int)

    # from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
    # conf_matrix = confusion_matrix(y_test, y_predicted)

    # print(f"F1 Score sul TEST SET: {f1_score(y_test, y_predicted)}")
    # print(f"Accuracy Score sul TEST SET: {accuracy_score(y_test, y_predicted)}")

    # sns.heatmap(conf_matrix, annot=True, fmt='d')
    # plt.show()
    
    # # import json
    # # with open(f'results/ADMM.json', 'r') as json_file:
    # #     json_content = json.load(json_file)
    
    # # for res in json_content:
        
    # #     if 'z' in res.keys():
    # #         print(res['agente'])
    # #         print(res['value_optimal_parameters'], res['value_cost_function'][-1])
    # #         print(res['z'])



