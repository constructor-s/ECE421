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

def drelu(x):
    return np.where(x > 0, 1, 0)

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

def dsoftmax(y):
    assert np.all(y > 0)
    assert np.all(y < 1)
    return y * (1 - y)

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

    K, N = t.shape
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
    Whidden = ran.normal(0.0, np.sqrt(2.0 / (input_layer_size + hidden_layer_size)), (input_layer_size, hidden_layer_size))
    bhidden = ran.normal(0.0, np.sqrt(2.0 / (input_layer_size + hidden_layer_size)), (hidden_layer_size, 1))
    Wout = ran.normal(0.0, np.sqrt(2.0 / (hidden_layer_size + output_layer_size)), (hidden_layer_size, output_layer_size))
    bout = ran.normal(0.0, np.sqrt(2.0 / (hidden_layer_size + output_layer_size)), (output_layer_size, 1))

    Whidden_v_old = np.zeros_like(Whidden)
    bhidden_v_old = np.zeros_like(bhidden)
    Wout_v_old = np.zeros_like(Wout)
    bout_v_old = np.zeros_like(bout)

    tt = time.perf_counter()
    n_epoches = 10
    for i in range(n_epoches):

        Shidden = computeLayer(trainData, Whidden, bhidden)
        Xhidden = relu(Shidden)
        Sout = computeLayer(Xhidden, Wout, bout)
        Xout = softmax(Sout)
        assert Xout.shape == newtrain.shape

        pred = np.argmax(Xout, 0)
        assert pred.shape == trainTarget.shape
        accuracy = np.count_nonzero(pred == trainTarget) * 1.0 / trainTarget.size

        print(i, accuracy, CE(newtrain, Xout))

        dl_dXout = gradCE(newtrain, Xout)
        dXout_dSout = dsoftmax(Xout)
        dSout_dWout = Xhidden

        dl_dSout = dl_dXout * dXout_dSout

        N = trainTarget.size

        # dl_dWout = 1.0 / N * (dl_dSout).dot(dSout_dWout.T)
        dl_dWout = dl_dSout.dot(dSout_dWout.T)
        dl_dWout = dl_dWout.T
        assert dl_dWout.shape == Wout.shape

        # dl_dbout = np.mean(dl_dSout, 1, keepdims=True)
        dl_dbout = np.sum(dl_dSout, 1, keepdims=True)
        assert dl_dbout.shape == bout.shape

        dSout_dXhidden = Wout
        dXhidden_dShidden = drelu(Shidden)

        dl_dShidden = dSout_dXhidden.dot(dl_dSout) * dXhidden_dShidden

        dShidden_dWhidden = trainData
        # dl_dWhidden = 1.0 / N * (dl_dShidden).dot(dShidden_dWhidden.T)
        dl_dWhidden = dl_dShidden.dot(dShidden_dWhidden.T)
        dl_dWhidden = dl_dWhidden.T
        assert dl_dWhidden.shape == Whidden.shape

        # dl_dbhidden = np.mean(dl_dShidden, 1, keepdims=True)
        dl_dbhidden = np.sum(dl_dShidden, 1, keepdims=True)
        assert dl_dbhidden.shape == bhidden.shape

        gamma = 0.99
        alpha = 1.0e-5

        Whidden_v_new = gamma * Whidden_v_old + alpha * dl_dWhidden
        bhidden_v_new = gamma * bhidden_v_old + alpha * dl_dbhidden
        Wout_v_new = gamma * Wout_v_old + alpha * dl_dWout
        bout_v_new = gamma * bout_v_old + alpha * dl_dbout

        Whidden -= Whidden_v_new
        bhidden -= bhidden_v_new
        Wout -= Wout_v_new
        bout -= bout_v_new

        Whidden_v_old = Whidden_v_new
        bhidden_v_old = bhidden_v_new
        Wout_v_old = Wout_v_new
        bout_v_old = bout_v_new

    print((time.perf_counter() - tt) * 1.0 / n_epoches, 'seconds per iter')


