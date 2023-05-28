import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from logistic_regression import sigmoid, cost_function
from sklearn.preprocessing import StandardScaler
from util import save_results_to_json
#Importo il dataset dal csv
data_training = pd.read_csv('test.csv')
x_tr = data_training.drop('diabetes', axis=1)
y_tr = data_training['diabetes']
#Normalizzo i dati
scaler = StandardScaler()
x_tr = scaler.fit_transform(x_tr)

# Aggiungo il bias
x_tr = np.hstack([np.ones([x_tr.shape[0],1]), x_tr])

#Varibili per il Gradient Descent
n_iterazioni = 100
n_parametri = 9
n_osservazioni = x_tr.shape[0]
learning_rate = [0.01, 0.001, 0.0001]
epsilon = .0000001

for step in learning_rate:
        #inizializzo una matrice in cui salvo i valore del parametri ad ogni iterazione
        pesi_parametri = np.zeros([n_parametri, n_iterazioni])
        #inizializzo i primi parametri da dare all'algoritmo in maniere casuale
        pesi_parametri[:,0] = np.ones(9)
        #creo una lista in cui vado a salvare il valore della funzione costo ad ogni iterazione
        valore_funzione_costo = []
        #calcolo il valore della funzione costo con i primi parametri casuali
        valore_funzione_costo.append(cost_function(x_tr, y_tr, pesi_parametri[:,0]))

        for i in range(0, n_iterazioni-1):
                # calcolo il valore della sigmoide
                h = sigmoid(np.matmul(x_tr, pesi_parametri[:,i]))
                #calcolo il gradiente
                gradient = np.matmul(x_tr.T, (h-y_tr)) / n_osservazioni
                #aggiorno i pesi
                pesi_parametri[:,i+1] = pesi_parametri[:,i] - step * gradient

                costo = cost_function(x_tr, y_tr, pesi_parametri[:,i+1])
                print(costo)
                valore_funzione_costo.append(costo)

                # if abs(valore_funzione_costo[i] - valore_funzione_costo[i+1]) < epsilon:
                #         print("Esco dal while poichè la funzione costo varia molto poco")
                # break

        dictionary = {
                'step': step,
                'value_cost_function': valore_funzione_costo,
                'value_optimal_parameters': pesi_parametri[:, n_iterazioni-1].tolist()
        }
        save_results_to_json(dictionary, 'gradient_descent')

        


y_predicted = sigmoid(np.dot(x_tr,pesi_parametri[:,n_iterazioni-1])) > 0.5
print(f"x_tr {x_tr.shape} pesi_parametri {pesi_parametri[:,n_iterazioni-1].shape}")
y_predicted = y_predicted.astype(int)

from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
conf_matrix = confusion_matrix(y_tr, y_predicted)

print(f"F1 Score sul TEST SET: {f1_score(y_tr, y_predicted)}")
print(f"Accuracy Score sul TEST SET: {accuracy_score(y_tr, y_predicted)}")

sns.heatmap(conf_matrix, annot=True, fmt='d')
plt.show()