import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import time
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Load the data
def loadData():
    with np.load("notMNIST.npz") as data:
        Data, Target = data["images"], data["labels"]
        np.random.seed(521)
        randIndx = np.arange(len(Data))
        np.random.shuffle(randIndx)
        Data = Data[randIndx] / 255.0
        Target = Target[randIndx]
        trainData, trainTarget = Data[:10000], Target[:10000]
        validData, validTarget = Data[10000:16000], Target[10000:16000]
        testData, testTarget = Data[16000:], Target[16000:]
    return trainData, validData, testData, trainTarget, validTarget, testTarget

# Implementation of a neural network using only Numpy - trained using gradient descent with momentum
def convertOneHot(trainTarget, validTarget, testTarget):
    newtrain = np.zeros((trainTarget.shape[0], 10))
    newvalid = np.zeros((validTarget.shape[0], 10))
    newtest = np.zeros((testTarget.shape[0], 10))

    for item in range(0, trainTarget.shape[0]):
        newtrain[item][trainTarget[item]] = 1
    for item in range(0, validTarget.shape[0]):
        newvalid[item][validTarget[item]] = 1
    for item in range(0, testTarget.shape[0]):
        newtest[item][testTarget[item]] = 1
    return newtrain, newvalid, newtest


def shuffle(trainData, trainTarget):
    np.random.seed(421)
    randIndx = np.arange(len(trainData))
    target = trainTarget
    np.random.shuffle(randIndx)
    data, target = trainData[randIndx], target[randIndx]
    return data, target


def relu(x):
    """
    >>> %timeit np.where(x>0, x, 0)
    145 µs ± 12.9 µs per loop (mean ± std. dev. of 7 runs, 10000 loops each)
    >>> %timeit x * (x>0)
    190 µs ± 2.78 µs per loop (mean ± std. dev. of 7 runs, 1000 loops each)
    >>> %timeit np.maximum(x, 0)
    983 µs ± 5.81 µs per loop (mean ± std. dev. of 7 runs, 1000 loops each)

    :param x:
    :return:
    """
    return np.where(x > 0, x, 0)

def softmax(x):
    """

    :param x:
    :return:
    """
    N = x.shape[1]
    assert x.shape == (10, N)
    ex = np.exp(x)
    ret = ex / np.sum(ex, 1, keepdims=True)
    assert ret.shape == (10, N)
    return ret

def computeLayer(X, W, b):
    """

    :param X: input, m_in x N
    :param W: weight, m_in x n_out
    :param b: bias, n_out x N
    :return:
    """
    X = np.atleast_2d(X)
    W = np.atleast_2d(W)
    b = np.asarray(b).reshape((-1, 1))

    m_in = X.shape[0]
    N = X.shape[1]
    n_out = W.shape[1]
    assert X.shape == (m_in, N)
    assert W.shape == (m_in, n_out)

    ret = X.T.dot(W)

    ret = ret.T
    assert ret.shape == (n_out, N)

    ret += b
    assert ret.shape == (n_out, N)

    return ret

def CE(target, prediction):
    """

    :param target:
    :param prediction:
    :return:
    """
    t = np.atleast_2d(target)
    s = np.atleast_2d(prediction)
    assert t.shape == s.shape

    K, N = t.shape
    assert K == 10

    logs = np.log(s)
    tlogs = t * logs

    averagece = - 1.0 / N * np.sum(tlogs)

    return float(averagece)

def gradCE(target, prediction):
    """

    :param target:
    :param prediction:
    :return:
    """
    t = np.atleast_2d(target)
    s = np.atleast_2d(prediction)
    assert t.shape == s.shape

    N, K = t.shape
    assert K == 10

    ret = - 1.0 / N * t / s
    assert ret.shape == t.shape

    return ret


if __name__ == '__main__':
    trainData, validData, testData, trainTarget, validTarget, testTarget = loadData()
    trainData = trainData.reshape((trainData.shape[0], -1)).T
    validData = validData.reshape((validData.shape[0], -1)).T
    testData = testData.reshape((testData.shape[0], -1)).T
    newtrain, newvalid, newtest = convertOneHot(trainTarget, validTarget, testTarget)
    newtrain = newtrain.T
    newvalid = newvalid.T
    newtest = newtest.T

    input_layer_size = trainData.shape[0]
    hidden_layer_size = 1000
    output_layer_size = newtrain.shape[0]

    ran = np.random.RandomState(421)
    W0 = ran.normal(0.0, 2.0 / (input_layer_size + hidden_layer_size), (input_layer_size, hidden_layer_size))
    b0 = ran.normal(0.0, 2.0 / (input_layer_size + hidden_layer_size), (hidden_layer_size, 1))
    W1 = ran.normal(0.0, 2.0 / (hidden_layer_size + output_layer_size), (hidden_layer_size, output_layer_size))
    b1 = ran.normal(0.0, 2.0 / (hidden_layer_size + output_layer_size), (output_layer_size, 1))

    X1 = computeLayer(trainData, W0, b0)
    S1 = relu(X1)
    X2 = computeLayer(S1, W1, b1)
    S2 = softmax(X2)
    assert S2.shape == newtrain.shape

    pred = np.argmax(S2, 0)
    assert pred.shape == trainTarget.shape
    accuracy = np.count_nonzero(pred == trainTarget) * 1.0 / trainTarget.size
    print(accuracy)

    print(CE(newtrain, S2))


