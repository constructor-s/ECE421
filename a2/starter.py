import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import time
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from matplotlib import pyplot as plt

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
    # ret = np.where(x > 0, x, 0)
    ret = x * (x > 0)
    return ret

# def drelu(x):
#     return np.where(x > 0, 1, 0)

def softmax(x):
    """
    :param x:
    :return:
    """
    N = x.shape[1]
    # assert x.shape == (10, N)  # TODO: remove
    x = x - np.max(x, 0, keepdims=True)  # Numerically stable version of softmax

    ex = np.exp(x)
    assert np.all(np.isfinite(ex))
    # assert np.count_nonzero(ex) == ex.size

    denom = np.sum(ex, 0, keepdims=True)
    assert denom.shape == (1, N)

    ret = ex / denom
    assert ret.shape == x.shape
    assert np.all(np.isfinite(ret))

    return ret

# def dsoftmax(y):
#     """
#
#     :param y: = softmax(x)
#     :return:
#     """
#     assert np.all(y > 0)
#     assert np.all(y < 1)
#
#     N = y.shape[1]
#     m = 10  # TODO: remove
#     assert y.shape == (m, N)
#
#     # def softmax_jacob(yy):
#     #     # yy = np.asarray(yy)
#     #     # assert np.squeeze(yy).ndim == 1
#     #     m = yy.size
#     #     ret = np.where(np.eye(m, m, dtype=np.bool), yy * (1 - yy), yy.reshape((m, 1)).dot(yy.reshape((1, m))))
#     #     # ret = np.eye(m, m) * (yy * (1-yy)) + (1 - np.eye(m, m)) * yy.reshape((m, 1)).dot(yy.reshape((1, m)))
#     #     # assert np.allclose(ret, np.eye(m, m) * (yy * (1-yy)) + (1 - np.eye(m, m)) * yy.reshape((m, 1)).dot(yy.reshape((1, m))))
#     #     return ret
#     #
#     # ret0 = np.apply_along_axis(softmax_jacob, axis=0, arr=y)
#
#     ret = np.einsum('ik,jk->ijk', -y, y)
#     assert ret.shape == (m, m, N)
#
#     # for i in range(N):
#     #     np.fill_diagonal(ret[:, :, i], y[:, i] * (1 - y[:, i]))
#
#     diag = np.einsum('ik,jk->ijk', y, 1 - y)
#
#     mask = np.moveaxis(np.tile(np.eye(m, dtype=np.bool), (N, 1, 1)), 0, 2)
#     assert mask.shape == diag.shape
#
#     ret = np.where(mask, diag, ret)
#
#     # assert np.allclose(ret0, ret)
#
#     # m, N = y.shape
#     # assert m == 10
#     #
#     # ret = [np.eye(m, m) * (yy * (1-yy)) + (1 - np.eye(m, m)) * yy.reshape((m, 1)).dot(yy.reshape((1, m))) for yy in y.T]
#     # ret = np.asarray(ret)
#
#     # ret = np.eye(m, m) * (y * (1-y)) + (1 - np.eye(m, m)) * y.dot(y.T)
#
#     return ret

def computeLayer(X, W, b):
    """
    :param X: input, m_in x N
    :param W: weight, m_in x n_out
    :param b: bias, n_out x N
    :return:
    """
    # m_in = X.shape[0]
    # N = X.shape[1]
    # n_out = W.shape[1]
    # assert X.shape == (m_in, N)
    # assert W.shape == (m_in, n_out)
    # assert b.shape == (n_out, 1)

    # ret = X.T.dot(W).T
    # assert ret.shape == (n_out, N)

    # ret += b

    ret = W.T.dot(X) + b
    # assert ret.shape == (n_out, N)

    return ret

def CE(target, prediction):
    """
    :param target:
    :param prediction:
    :return:
    """
    t = np.asarray(target)
    s = np.asarray(prediction)
    assert t.shape == s.shape
    # assert tuple(np.sort(np.unique(t))) == (0, 1)

    K, N = t.shape
    # print(N)
    # assert K == 10  # TODO: Remove

    # logs = np.log(s)
    # tlogs = t * logs
    tlogs = np.zeros_like(t)
    tlogs[t == 1] = np.log(s[t == 1])
    print(tlogs.min())

    averagece = - 1.0 / N * np.sum(tlogs)
    averagece = float(averagece)
    assert np.isfinite(averagece)

    return averagece

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
    # assert K == 10

    ret = - 1.0 / N * t / s
    assert ret.shape == t.shape

    return ret

def nn(x, weights, biases, dropout):

    #x = tf.reshape(x, shape=[-1, 28, 28, 1])

    # Convolution Layer
    conv = tf.nn.conv2d(x, W, filter = [1,1,3,32] , strides=[1, 1], padding='SAME') #RGB:3??
    re_lu = tf.nn.relu(x)
    
    #batch normalization
        #offset: often denoted beta in equations
        #scale: often denoted gamma in equations
        #variance_epsilon: a small float number to avoid dividing by 0
    tf.nn.batch_normalization(x, mean, variance, offset, scale, variance_epsilon, name=None)    
    
    # Max Pooling
    conv = maxpool2d(conv, k=2)
    
    #flatten layer
    conv_ft = tf.layers.flatten(conv)
    
    # Fully connected layer
    W1 = tf.get_variable("weights1", shape=[784, 10], initializer=tf.glorot_uniform_initializer())
    b1 = tf.get_variable("bias1", shape=[10], initializer=tf.constant_initializer(0.1))
    
    x_drop = tf.nn.dropout(x, dropout)
    fc1 = tf.nn.relu(tf.matmul(x_drop, W1) + b1)
    
    W2 = tf.get_variable("weights2", shape=[10, 10], initializer=tf.glorot_uniform_initializer())                
    b2 = tf.get_variable("bias2", shape=[10], initializer=tf.constant_initializer(0.1))
    
    # Apply Dropout
    fc1_drop = tf.nn.dropout(fc1, dropout)
    fc2 = tf.nn.relu(tf.matmul(fc1_drop,W2) + b2)
    
    #logits = tf.nn.relu(tf.matmul(x, W) + b)
    #labels = tf.placeholder(tf.float32, [None, 10])
    
    #cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels)

    # Output, class prediction
    out = tf.add(tf.matmul(fc1, weights['out']), biases['out'])
    return out


def forward(trainData, Whidden, bhidden, Wout, bout):
    Shidden = computeLayer(trainData, Whidden, bhidden)
    Xhidden = relu(Shidden)
    Sout = computeLayer(Xhidden, Wout, bout)
    Xout = softmax(Sout)

    assert np.all(np.isfinite(Shidden))
    assert np.all(np.isfinite(Xhidden))
    assert np.all(np.isfinite(Sout))
    assert np.all(np.isfinite(Xout))

    # assert Xout.shape == newtrain.shape
    # Accuracy and loss calculation
    pred = np.argmax(Xout, 0)
    return Shidden, Xhidden, Sout, Xout, pred


if __name__ == '__main__':
# def main():
    #%% Initialize dataset
    trainData, validData, testData, trainTarget, validTarget, testTarget = loadData()

    # trainData = trainData[trainTarget < 4, :]
    # validData = validData[validTarget < 4, :]
    # testData = testData[testTarget < 4, :]
    # trainTarget = trainTarget[trainTarget < 4]
    # validTarget = validTarget[validTarget < 4]
    # testTarget = testTarget[testTarget < 4]

    trainData = trainData.reshape((trainData.shape[0], -1)).T - 0.5
    validData = validData.reshape((validData.shape[0], -1)).T - 0.5
    testData = testData.reshape((testData.shape[0], -1)).T - 0.5

    newtrain, newvalid, newtest = convertOneHot(trainTarget, validTarget, testTarget)

    newtrain = newtrain.T
    newvalid = newvalid.T
    newtest = newtest.T

    # trainTarget = trainTarget[0:1000]
    # validTarget = validTarget[0:1000]
    # testTarget = testTarget[0:1000]
    # trainData = trainData[:, 0:1000]
    # validData = validData[:, 0:1000]
    # testData = testData[:, 0:1000]
    # newtrain = newtrain[:, 0:1000]
    # newvalid = newvalid[:, 0:1000]
    # newtest = newtest[:, 0:1000]

    # %% Initialize weights
    input_layer_size = trainData.shape[0]
    hidden_layer_size = 500
    output_layer_size = newtrain.shape[0]
    N = trainTarget.size

    ran = np.random.RandomState(421)
    Whidden = ran.normal(0.0, np.sqrt(2.0 / (input_layer_size + hidden_layer_size)), (input_layer_size, hidden_layer_size))
    bhidden = np.zeros((hidden_layer_size, 1))
    Wout = ran.normal(0.0, np.sqrt(2.0 / (hidden_layer_size + output_layer_size)), (hidden_layer_size, output_layer_size))
    bout = np.zeros((output_layer_size, 1))

    Whidden_v_old = np.zeros_like(Whidden)
    bhidden_v_old = np.zeros_like(bhidden)
    Wout_v_old = np.zeros_like(Wout)
    bout_v_old = np.zeros_like(bout)

    tt = time.perf_counter()
    gamma = 0.5
    alpha = 0.1

    #%% Training
    n_epoches = 200
    results = []
    for i in range(n_epoches):
        # Forward pass
        Shidden, Xhidden, Sout, Xout, pred = forward(trainData, Whidden, bhidden, Wout, bout)
        assert pred.shape == trainTarget.shape
        accuracy = np.count_nonzero(pred == trainTarget) * 1.0 / trainTarget.size

        ce = CE(newtrain, Xout)
        print(i, accuracy, ce)

        _, _, _, validxout, validpred = forward(validData, Whidden, bhidden, Wout, bout)
        validaccuracy = np.count_nonzero(validpred == validTarget) * 1.0 / validTarget.size
        validce = CE(newvalid, validxout)
        print(i, validaccuracy, validce)

        results.append((accuracy, ce, validaccuracy, validce))

        # Back prop
        # dl_dXout = gradCE(newtrain, Xout)
        # assert dl_dXout.shape == (output_layer_size, N)
        #
        # dXout_dSout = dsoftmax(Xout)
        # assert dXout_dSout.shape == (output_layer_size, output_layer_size, N)
        #
        # dl_dSout = np.einsum('ijk,jk->ik', dXout_dSout, dl_dXout)
        dl_dSout = 1.0 / N * np.where(newtrain, Xout - 1, Xout)
        assert dl_dSout.shape == (output_layer_size, N)

        dl_dWout = Xhidden.dot(dl_dSout.T)
        assert dl_dWout.shape == (hidden_layer_size, output_layer_size)
        assert dl_dWout.shape == Wout.shape

        dl_dbout = np.sum(dl_dSout, 1, keepdims=True)
        assert dl_dbout.shape == (output_layer_size, 1)
        assert dl_dbout.shape == bout.shape


        # dl_dXhidden = Wout.dot(dl_dSout)
        # assert dl_dXhidden.shape == (hidden_layer_size, N)
        #
        # dl_dShidden = dl_dXhidden * drelu(dl_dXhidden)
        dl_dShidden = np.where(Shidden > 0, Wout.dot(dl_dSout), 0)
        assert dl_dShidden.shape == (hidden_layer_size, N)

        dl_dWhidden = trainData.dot(dl_dShidden.T)
        assert dl_dWhidden.shape == (input_layer_size, hidden_layer_size)
        assert dl_dWhidden.shape == Whidden.shape

        dl_dbhidden = np.sum(dl_dShidden, 1, keepdims=True)
        assert dl_dbhidden.shape == (hidden_layer_size, 1)
        assert dl_dbhidden.shape == bhidden.shape

        assert np.all(np.isfinite(dl_dWhidden))
        assert np.all(np.isfinite(dl_dbhidden))
        assert np.all(np.isfinite(dl_dWout))
        assert np.all(np.isfinite(dl_dbout))

        Whidden_v_new = gamma * Whidden_v_old + (1-gamma) * dl_dWhidden
        bhidden_v_new = gamma * bhidden_v_old + (1-gamma) * dl_dbhidden
        Wout_v_new = gamma * Wout_v_old + (1-gamma) * dl_dWout
        bout_v_new = gamma * bout_v_old + (1-gamma) * dl_dbout

        Whidden -= alpha * Whidden_v_new
        bhidden -= alpha * bhidden_v_new
        Wout -= alpha * Wout_v_new
        bout -= alpha * bout_v_new

        assert np.all(np.isfinite(Whidden))
        assert np.all(np.isfinite(bhidden))
        assert np.all(np.isfinite(Wout))
        assert np.all(np.isfinite(bout))

        Whidden_v_old = Whidden_v_new
        bhidden_v_old = bhidden_v_new
        Wout_v_old = Wout_v_new
        bout_v_old = bout_v_new

        print((time.perf_counter() - tt) * 1.0 / (i+1), 'seconds per iter')

        if i % 20 == 0:
            fig, ax = plt.subplots(2, 1)
            ax[0].plot(np.asarray(results)[:, [0, 2]])
            ax[1].plot(np.asarray(results)[:, [1, 3]])
            fig.show()


    fig, ax = plt.subplots(2, 1)
    ax[0].plot(np.asarray(results)[:, [0,2]])
    ax[1].plot(np.asarray(results)[:, [1,3]])
    fig.show()

# from line_profiler import LineProfiler
# lp = LineProfiler()
# lp.add_function(computeLayer)
# lp.add_function(forward)
# lp_wrapper = lp(main)
# lp_wrapper()
# lp.print_stats()
