import numpy as np
import pandas as pd
import neural_network_backbone as nnb
from sklearn.metrics import accuracy_score 

SILENT = False
SEED = 10101
EPOCHS = 300
LEARNING_RATE = 0.1
INPUTS_DIRECTORY = './inputs/classification/'
TRAIN_FILE = 'data.simple.train.100.csv'
TEST_FILE = 'data.simple.test.100.csv'

def get_progress(Y_hat, Y):
    cost = get_cost_value(Y_hat, Y)
    accuracy = get_accuracy_value(Y_hat, Y)
    return "cost: {:.5f} - accuracy: {:.5f}".format(cost, accuracy)

def one_hot(Y, num_classes):
    return np.squeeze(np.eye(num_classes)[Y.reshape(-1)])

def convert_prob_into_class(probs):
    return np.array([[1. if prob == max(v) else 0. for prob in v] for v in probs]).reshape(probs.shape)

def get_accuracy_value(Y_hat, Y):
    Y = one_hot(Y, n_outputs)
    Y_hat_ = convert_prob_into_class(Y_hat.T)
    return accuracy_score(Y, Y_hat_)

def get_cost_value(Y_hat, Y, derivative = False):
    Y = one_hot(Y, n_outputs)
    if not derivative:
        eps = 1e-15
        Y_hat = np.clip(Y_hat, eps, 1. - eps)
        return -np.mean(Y * np.log(Y_hat.T) + (1. - Y) * np.log(1. - Y_hat.T))
    else:
        return Y_hat.T - Y
    
# dataset_train = pd.read_csv(INPUTS_DIRECTORY  + TRAIN_FILE, sep=',').values
# dataset_test = pd.read_csv(INPUTS_DIRECTORY + TEST_FILE, sep=',').values

np.random.seed(SEED)
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
X, y = make_blobs(n_samples=1000, centers=3, n_features=2)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

n_inputs = 2
n_outputs = 3

# X_train = dataset_train[:,0:2]
# y_train = dataset_train[:,2].astype(int) - 1

# X_test = dataset_test[:,0:2]
# y_test = dataset_test[:,2].astype(int) - 1


network_layers = [
    {"nodes": n_inputs},
    {"nodes": 5, "activation": nnb.relu},
    {"nodes": n_outputs, "activation": nnb.softmax}
]

nnb.SILENT = SILENT
nnb.COST_FUNC = get_cost_value
nnb.PROGRESS_FUNC = get_progress
params_values = nnb.train(X_train, y_train.reshape((y_train.shape[0], 1)), 
                          network_layers, EPOCHS, LEARNING_RATE, SEED)
Y_test_hat, _ = nnb.full_forward_propagation(np.transpose(X_test), params_values, network_layers)
print("Test set: " + get_progress(Y_test_hat, np.transpose(y_test.reshape((y_test.shape[0], 1)))))