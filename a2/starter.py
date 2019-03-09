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
    ex = np.exp(x)
    return ex / np.sum(ex)

def computeLayer(X, W, b):
    """

    :param X: input, m x 1
    :param W: weight, m x n
    :param b: bias, n x 1
    :return:
    """
    X = np.asarray(X).reshape((-1, 1))
    W = np.atleast_2d(W)
    b = np.asarray(b).reshape((-1, 1))

    m = X.size
    n = W.shape[1]
    assert X.shape == (m, 1)
    assert W.shape[0] == m

    ret = X.T.dot(W)

    ret = ret.T
    assert ret.shape == (n, 1)

    ret += b
    assert ret.shape == (n, 1)

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

    N, K = t.shape
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
    newtrain, newvalid, newtest = convertOneHot(trainTarget, validTarget, testTarget)

    def q1():
        pass
