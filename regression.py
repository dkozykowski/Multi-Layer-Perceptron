import numpy as np
import pandas as pd
import neural_network_backbone as nnb
from sklearn.metrics import mean_absolute_error

SILENT = False
SEED = 10101
EPOCHS = 10000
LEARNING_RATE = 0.1
INPUTS_DIRECTORY = './inputs/regression/'
TRAIN_FILE = 'data.activation.train.100.csv'
TEST_FILE = 'data.activation.test.100.csv'

def get_accuracy_value(Y_hat, Y):
    return 0

def get_cost_value(Y_hat, Y):
    return np.sqrt(np.sum(np.square(Y - Y_hat)) / len(Y))

dataset_train = pd.read_csv(INPUTS_DIRECTORY  + TRAIN_FILE, sep=',').values
dataset_test = pd.read_csv(INPUTS_DIRECTORY + TEST_FILE, sep=',').values

n_inputs = len(dataset_train[0]) - 1
n_outputs = 1

X_train = dataset_train[:,0:1]
y_train = dataset_train[:,1]

X_test = dataset_test[:,0:1]
y_test = dataset_test[:,1]


network_layers = [
    {"nodes": n_inputs},
    {"nodes": 5, "activation": nnb.relu},
    {"nodes": 1, "activation": nnb.linear},
]

nnb.SILENT = SILENT
nnb.ACCURACY_FUNC = get_accuracy_value
nnb.COST_FUNC = get_cost_value
params_values = nnb.train(np.transpose(X_train), np.transpose(y_train.reshape((y_train.shape[0], 1))), 
                          network_layers, EPOCHS, LEARNING_RATE, SEED)
Y_test_hat, _ = nnb.full_forward_propagation(np.transpose(X_test), params_values, network_layers)
acc_test = get_accuracy_value(Y_test_hat, np.transpose(y_test.reshape((y_test.shape[0], 1))))
print("Test set accuracy: {:.2f}".format(acc_test))