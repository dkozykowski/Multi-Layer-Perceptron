import numpy as np
import pandas as pd
import neural_network_backbone as nnb
from sklearn.metrics import r2_score

SILENT = False
SEED = 10101
EPOCHS = 1000
LEARNING_RATE = 0.3
INPUTS_DIRECTORY = './inputs/regression/'
TRAIN_FILE = 'data.activation.train.100.csv'
TEST_FILE = 'data.activation.test.100.csv'

def get_progress(Y_hat, Y):
    cost = get_cost_value(Y_hat, Y)
    mean_error = np.mean(np.abs(Y_hat - Y))
    return "cost: {:.5f} - mean error: {:.5f}".format(cost, mean_error)

def get_cost_value(Y_hat, Y, derivative = False):
    if not derivative:
        m = Y.shape[1]
        return 1 / m * np.sum(np.square(Y - Y_hat))
    else:
        return -2 * (Y - Y_hat)

dataset_train = pd.read_csv(INPUTS_DIRECTORY  + TRAIN_FILE, sep=',').values
dataset_test = pd.read_csv(INPUTS_DIRECTORY + TEST_FILE, sep=',').values

n_inputs = len(dataset_train[0]) - 1
n_outputs = 1

X_train = dataset_train[:,0:1]
y_train = dataset_train[:,1]
miniY = np.min(y_train)
y_train -= miniY
maxiY = np.max(y_train)
y_train /= maxiY
y_train *= 2
y_train -= 1

X_test = dataset_test[:,0:1]
y_test = ((dataset_test[:,1] - miniY) / maxiY) * 2 - 1

network_layers = [
    {"nodes": n_inputs},
    {"nodes": 5, "activation": nnb.tanh},
    {"nodes": 1, "activation": nnb.linear},
]

nnb.SILENT = SILENT
nnb.COST_FUNC = get_cost_value
nnb.PROGRESS_FUNC = get_progress
params_values = nnb.train(np.transpose(X_train), np.transpose(y_train.reshape((y_train.shape[0], 1))), 
                          network_layers, EPOCHS, LEARNING_RATE, SEED)
Y_test_hat, _ = nnb.full_forward_propagation(np.transpose(X_test), params_values, network_layers)
print("Test set: " + get_progress(Y_test_hat, np.transpose(y_test.reshape((y_test.shape[0], 1)))))