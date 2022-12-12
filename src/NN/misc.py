import numpy as np


def R2(z_test,z_pred):
    """computes the mean squared error for a given prediction

    :z_test: array-like
    :z_pred: array-like
    :returns: R squared score

    """

    return 1 - np.mean((z_test - z_pred)**2)/np.mean((z_test - np.mean(z_test))**2)


# some of these are from lecture notes with modification
def MSE(z_test, z_pred):
    """computes the mean squared error for a given prediction

    :z_test: array-like
    :z_pred: array-like
    :returns: Mean squared error

    """

    return np.mean((z_test - z_pred)**2)

def MSE_prime(y_true, y_pred):
    return 2*(y_pred-y_true)/y_true.size

def cal_bias(z_test, z_pred):
    """computes the bias for a given prediction

    :z_test: array-like
    :z_pred: array-like
    :returns: bias

    """
    return np.mean( (z_test - np.mean(z_pred, axis=1, keepdims=True))**2 )

def cal_variance(z_pred):
    """computes the variance for a given prediction

    :z_test: array-like
    :z_pred: array-like
    :returns: variance

    """
    return np.mean( np.var(z_pred, axis=1, keepdims=True) )

def logistic_grad(y_true, y_pred):
    # since only paired with Sigmoid, error can be y-p
    result = y_pred-y_true
    return result

def accuracy_score(y,y_pred):
    n = len(y)
    I = 0
    for i in range(n):
        if y[i] == y_pred[i]:
            I += 1
    return I/n

def softmax(x):
    exponential = np.exp(x)
    return exponential / np.sum(exponential)

def softmax_grad(x,i,j):
    sm = softmax(x)
    if i == j:
        return sm*(1-sm)
    return -sm**2
