import numpy as np

# https://stackoverflow.com/questions/38125319/python-divide-by-zero-encountered-in-log-logistic-regression
epsilon = 0.00001

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

#La funzione costo cambia nel caso in cui sto utilizzando ADMM in cui calcolo il costo per ogni agente
def cost_function(x, y, pesi_parametri, n_agenti=0):
    
    if n_agenti == 0:
        h = sigmoid(np.matmul(x, pesi_parametri))
        m = x.shape[0]
        # h = sigmoid(x @ pesi_parametri)
        cost = (np.dot(-y.T, np.log(h + epsilon)) - np.dot((1 -y.T), np.log(1 - h + epsilon))) / m
        return cost
        
    cost = np.zeros([n_agenti,1])
    for i in range(0, n_agenti):
        dati_per_agente = int(x.shape[0]/n_agenti)
        x_i = x[dati_per_agente*i:dati_per_agente*(i+1), :]
        y_i = y[dati_per_agente*i:dati_per_agente*(i+1)]

        cost[i,:] = cost_function(x_i, y_i, pesi_parametri[:,i])
    return cost.reshape(4)


#Funzione che calcola l'Hessiana
def compute_hessian(x, theta):
    s = sigmoid(np.matmul(x,theta))
    diag = np.diag(s * (1 - s))
    H = np.dot(np.dot(x.T, diag), x)
    return H