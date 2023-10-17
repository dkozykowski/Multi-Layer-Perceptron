import numpy as np
import shutil
import os
import pickle

SILENT = True
COST_FUNC = None
PROGRESS_FUNC = None

def sigmoid(Z, derivative = False):
    sig = 1 / (1 + np.exp(-Z))
    if not derivative:
        return sig
    else:
        return sig * (1 - sig)

def relu(Z, derivative = False):
    if not derivative:
        return np.maximum(0, Z)
    else:
        dZ = np.array(Z, copy = True)
        dZ[Z <= 0] = 0
        dZ[Z > 0] = 1
        return dZ

def linear(Z, derivative = False):
    if not derivative:
        return np.array(Z, copy = True)
    else:
        return np.ones(shape=Z.shape) 
    
def tanh(Z, derivative = False):
    if not derivative:
        return np.tanh(Z)
    else:
        return 1.0 - np.tanh(Z)**2
    
def softmax(Z, derivative = False):
    if not derivative:
        return np.exp(Z) / sum(np.exp(Z))
    else:
        E = np.exp(Z)
        S = sum(E)
        return (S - E) * E / (S ** 2)

def init_layers(network_layers, seed):
    np.random.seed(seed)
    params_values = {}
    layers_number = len(network_layers)
    for i in range(1, layers_number):
        previous_layer_nodes = network_layers[i - 1]["nodes"]
        current_layer_nodes = network_layers[i]["nodes"]    
        params_values['W' + str(i)] = np.random.randn(current_layer_nodes, 
                                                      previous_layer_nodes) * 0.1
        params_values['b' + str(i)] = np.random.randn(current_layer_nodes, 1) * 0.1
    return params_values

def single_layer_forward_propagation(A_prev, W_curr, b_curr, activation_func):
    Z_curr = np.dot(W_curr, A_prev) + b_curr
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

def single_layer_backward_propagation(dA_curr, W_curr, b_curr, Z_curr, A_prev, activation_func):
    m = A_prev.shape[1]
    dZ_curr = np.multiply(activation_func(Z_curr, derivative = True), dA_curr)
    dW_curr = np.dot(dZ_curr, A_prev.T) / m
    db_curr = np.sum(dZ_curr, axis = 1, keepdims = True) / m
    dA_prev = np.dot(W_curr.T, dZ_curr)
    return dA_prev, dW_curr, db_curr

def full_backward_propagation(Y_hat, Y, memory, params_values, network_layers):
    grads_values = {}
    layers_number = len(network_layers)
    dA_prev = COST_FUNC(Y_hat, Y, True).T
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

def train(X, Y, network_layers, epochs, learning_rate, seed):
    params_values = init_layers(network_layers, seed)
    for i in range(epochs):
        Y_hat, cashe = full_forward_propagation(X, params_values, network_layers)
        grads_values = full_backward_propagation(Y_hat, Y, cashe, params_values, network_layers)
        params_values = update(params_values, grads_values, network_layers, learning_rate)
        if not SILENT and i % (epochs / 50) == 0:
            print("Iteration: {:05} - ".format(i) + PROGRESS_FUNC(Y_hat, Y))
    return params_values

def save_model(network_layers, params_values, filename):
    network_layers_filename = filename + "/network_layers.npy"
    params_values_filename = filename + "/params_values.npy"
    os.makedirs(os.path.dirname(network_layers_filename))
    with open(network_layers_filename, 'wb') as f:
        pickle.dump(network_layers, f)
    with open(params_values_filename, 'wb') as f:
        pickle.dump(params_values, f)
    shutil.make_archive(filename, 'zip', filename)
    cleanup(filename)

def load_model(filename):
    network_layers_filename = filename + "/network_layers.npy"
    params_values_filename = filename + "/params_values.npy"
    shutil.unpack_archive(filename + ".zip", filename, "zip")
    with open(network_layers_filename, 'rb') as f:
        network_layers = pickle.load(f)
    with open(params_values_filename, 'rb') as f:
       params_values = pickle.load(f)
    cleanup()
    return network_layers, params_values
    
def cleanup(filename):
    network_layers_filename = filename + "/network_layers.npy"
    params_values_filename = filename + "/params_values.npy"
    os.remove(network_layers_filename)
    os.remove(params_values_filename)
    os.rmdir(filename)
