import numpy as np


# activation function
def relu(inputs):
    return np.maximum(inputs, 0)

# output probability distribution function
def softmax(inputs):
    exp = np.exp(inputs)
    return exp/np.sum(exp, axis = 1, keepdims = True)

def softmax2(inputs):
    t = inputs
    o = inputs/np.sum(t, axis = 1, keepdims = True)
    o = np.maximum(o, 0.01)
    return o

def softmax3(inputs):
    exp = np.power(2, inputs)
    return exp/np.sum(exp, axis = 1, keepdims = True)

# loss
def cross_entropy(inputs, y):
    indices = np.argmax(y, axis = 1).astype(int)
    probability = inputs[np.arange(len(inputs)), indices] #inputs[0, indices]
    log = np.log(probability)
    loss = -1.0 * np.sum(log) / len(log)
    return loss

def error_function(inputs, y):
    # Average of the square of prediction - actual
    # Each parameter, inputs and y are vectors
    # cost = np.sum(np.power((inputs - y), 4)) / y.shape[0]
    # cost = np.sum(np.abs(inputs-y)) / y.shape[0]
    # cost = np.sum(np.square(inputs - y)) / y.shape[0]
    cost = 0.1
    return cost

# L2 regularization
def L2_regularization(la, weight1, weight2):
    weight1_loss = 0.5 * la * np.sum(weight1 * weight1)
    weight2_loss = 0.5 * la * np.sum(weight2 * weight2)
    return weight1_loss + weight2_loss
