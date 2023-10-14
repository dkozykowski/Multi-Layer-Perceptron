import numpy as np
import pandas as pd
import neural_network_backbone as nnb

SILENT = False
SEED = 10101
EPOCHS = 300
LEARNING_RATE = 0.1
INPUTS_DIRECTORY = './inputs/classification/'
TRAIN_FILE = 'data.simple.train.100.csv'
TEST_FILE = 'data.simple.test.100.csv'

def convert_prob_into_class(probs):
    probs_ = np.copy(probs)
    probs_[probs_ > 0.5] = 1
    probs_[probs_ <= 0.5] = 0
    return probs_

def get_accuracy_value(Y_hat, Y):
    Y_hat_ = convert_prob_into_class(Y_hat)
    return (Y_hat_ == Y).all(axis=0).mean()

dataset_train = pd.read_csv(INPUTS_DIRECTORY  + TRAIN_FILE, sep=',').values
dataset_test = pd.read_csv(INPUTS_DIRECTORY + TEST_FILE, sep=',').values

n_inputs = len(dataset_train[0]) - 1
n_outputs = len(set([row[-1] for row in dataset_train]))

X_train = dataset_train[:,0:2]
y_train = dataset_train[:,2].astype(int) - 1

X_test = dataset_test[:,0:2]
y_test = dataset_test[:,2].astype(int) - 1

network_layers = [
    {"nodes": n_inputs},
    {"nodes": 5, "activation": "relu"},
    {"nodes": 1, "activation": "sigmoid"},
]

nnb.SILENT = SILENT
nnb.ACCURACY_FUNC = get_accuracy_value
params_values = nnb.train(np.transpose(X_train), np.transpose(y_train.reshape((y_train.shape[0], 1))), 
                          network_layers, EPOCHS, LEARNING_RATE, SEED)
Y_test_hat, _ = nnb.full_forward_propagation(np.transpose(X_test), params_values, network_layers)
acc_test = get_accuracy_value(Y_test_hat, np.transpose(y_test.reshape((y_test.shape[0], 1))))
print("Test set accuracy: {:.2f}".format(acc_test))