import numpy as np

# https://stackoverflow.com/questions/38125319/python-divide-by-zero-encountered-in-log-logistic-regression
epsilon = 0.00001

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def cost_function(x, y, pesi_parametri):
    h = sigmoid(np.matmul(x, pesi_parametri))
    m = x.shape[0]
    # h = sigmoid(x @ pesi_parametri)
    cost = (np.matmul(-y.T, np.log(h + epsilon)) - np.matmul((1 -y.T), np.log(1 - h) + epsilon))/ m
    return cost

#Funzione che calcola l'Hessiana
def compute_hessian(x, theta):
    s = sigmoid(np.matmul(x,theta))
    diag = np.diag(s * (1 - s))
    H = np.dot(np.dot(x.T, diag), x)
    return H