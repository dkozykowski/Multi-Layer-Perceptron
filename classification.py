import numpy as np
import pandas as pd

SEED = 10101
EPOCHS = 10000
LEARNING_RATE = 0.01
INPUTS_DIRECTORY = './inputs/classification/'
TRAIN_FILE = 'data.simple.train.100.csv'
TEST_FILE = 'data.simple.test.100.csv'

def sigmoid(Z):
    return 1 / (1 + np.exp(-Z))

def relu(Z):
    return np.maximum(0, Z)

def sigmoid_backward(dA, Z):
    sig = sigmoid(Z)
    return dA * sig * (1 - sig)

def relu_backward(dA, Z):
    dZ = np.array(dA, copy = True)
    dZ[Z <= 0] = 0
    return dZ

def init_layers(network_layers):
    params_values = {}
    layers_number = len(network_layers)
    for i in range(1, layers_number):
        previous_layer_nodes = network_layers[i - 1]["nodes"]
        current_layer_nodes = network_layers[i]["nodes"]
        params_values['W' + str(i)] = np.random.randn(
            current_layer_nodes, previous_layer_nodes) * 0.1
        params_values['b' + str(i)] = np.random.randn(
            current_layer_nodes, 1) * 0.1
    return params_values

def single_layer_forward_propagation(A_prev, W_curr, b_curr, activation):
    Z_curr = np.dot(W_curr, A_prev) + b_curr
    if activation == "relu":
        activation_func = relu
    elif activation == "sigmoid":
        activation_func = sigmoid
    else:
        raise Exception('Non-supported activation function')
    return activation_func(Z_curr), Z_curr

def full_forward_propagation(X, params_values, network_layers):
    memory = {}
    A_curr = X
    layers_number = len(network_layers)
    for i in range(1, layers_number):
        A_prev = A_curr
        activ_function_curr = network_layers[i]["activation"]
        W_curr = params_values["W" + str(i)]
        b_curr = params_values["b" + str(i)]
        A_curr, Z_curr = single_layer_forward_propagation(A_prev, W_curr, b_curr, activ_function_curr)
        memory["A" + str(i - 1)] = A_prev
        memory["Z" + str(i)] = Z_curr
    return A_curr, memory

def get_cost_value(Y_hat, Y):
    m = Y_hat.shape[1]
    cost = -1 / m * (np.dot(Y, np.log(Y_hat).T) + np.dot(1 - Y, np.log(1 - Y_hat).T))
    return np.squeeze(cost)

def convert_prob_into_class(probs):
    probs_ = np.copy(probs)
    probs_[probs_ > 0.5] = 1
    probs_[probs_ <= 0.5] = 0
    return probs_

def get_accuracy_value(Y_hat, Y):
    Y_hat_ = convert_prob_into_class(Y_hat)
    return (Y_hat_ == Y).all(axis=0).mean()

def single_layer_backward_propagation(dA_curr, W_curr, b_curr, Z_curr, A_prev, activation="relu"):
    m = A_prev.shape[1]
    if activation == "relu":
        backward_activation_func = relu_backward
    elif activation == "sigmoid":
        backward_activation_func = sigmoid_backward
    else:
        raise Exception('Non-supported activation function')
    dZ_curr = backward_activation_func(dA_curr, Z_curr)
    dW_curr = np.dot(dZ_curr, A_prev.T) / m
    db_curr = np.sum(dZ_curr, axis=1, keepdims=True) / m
    dA_prev = np.dot(W_curr.T, dZ_curr)
    return dA_prev, dW_curr, db_curr

def full_backward_propagation(Y_hat, Y, memory, params_values, network_layers):
    grads_values = {}
    m = Y.shape[1]
    Y = Y.reshape(Y_hat.shape)
    dA_prev = - (np.divide(Y, Y_hat) - np.divide(1 - Y, 1 - Y_hat))
    layers_number = len(network_layers)
    for i in reversed(range(1, layers_number)):
        activ_function_curr = network_layers[i]["activation"]
        dA_curr = dA_prev
        A_prev = memory["A" + str(i - 1)]
        Z_curr = memory["Z" + str(i)]
        W_curr = params_values["W" + str(i)]
        b_curr = params_values["b" + str(i)]
        dA_prev, dW_curr, db_curr = single_layer_backward_propagation(
            dA_curr, W_curr, b_curr, Z_curr, A_prev, activ_function_curr)
        grads_values["dW" + str(i)] = dW_curr
        grads_values["db" + str(i)] = db_curr
    return grads_values

def update(params_values, grads_values, network_layers, learning_rate):
    layers_number = len(network_layers)
    for i in range(1, layers_number):
        params_values["W" + str(i)] -= learning_rate * grads_values["dW" + str(i)]        
        params_values["b" + str(i)] -= learning_rate * grads_values["db" + str(i)]
    return params_values

def train(X, Y, network_layers, epochs, learning_rate):
    params_values = init_layers(network_layers)
    accuracy_history = []
    
    for i in range(epochs):
        Y_hat, cashe = full_forward_propagation(X, params_values, network_layers)
        cost = get_cost_value(Y_hat, Y)
        accuracy = get_accuracy_value(Y_hat, Y)
        accuracy_history.append(accuracy)
        grads_values = full_backward_propagation(Y_hat, Y, cashe, params_values, network_layers)
        params_values = update(params_values, grads_values, network_layers, learning_rate)
        if i % (epochs / 50) == 0:
            print("Iteration: {:05} - cost: {:.5f} - accuracy: {:.5f}".format(i, cost, accuracy))
            
    return params_values


np.random.seed(SEED)
dataset_train = pd.read_csv(INPUTS_DIRECTORY  + TRAIN_FILE, sep=',').values
dataset_test = pd.read_csv(INPUTS_DIRECTORY + TEST_FILE, sep=',').values

n_inputs = len(dataset_train[0]) - 1
n_outputs = len(set([row[-1] for row in dataset_train]))

X_train = dataset_train[:,0:2]
y_train = dataset_train[:,2].astype(int) - 1

X_test = dataset_test[:,0:2]
y_test = dataset_test[:,2].astype(int) - 1


####################### TEMPORARY ONLY FOR TEST PURPOSES

N_SAMPLES = 1000
TEST_SIZE = 0.1
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
X, y = make_moons(n_samples = N_SAMPLES, noise=0.2, random_state=100)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=42)

#######################


network_layers = [
    {"nodes": 2},
    {"nodes": 25, "activation": "relu"},
    {"nodes": 50, "activation": "relu"},
    {"nodes": 50, "activation": "relu"},
    {"nodes": 25, "activation": "relu"},
    {"nodes": 1, "activation": "sigmoid"},
]

params_values = train(np.transpose(X_train), np.transpose(y_train.reshape((y_train.shape[0], 1))), network_layers, EPOCHS, LEARNING_RATE)
Y_test_hat, _ = full_forward_propagation(np.transpose(X_test), params_values, network_layers)
acc_test = get_accuracy_value(Y_test_hat, np.transpose(y_test.reshape((y_test.shape[0], 1))))
print("Test set accuracy: {:.2f}".format(acc_test))
