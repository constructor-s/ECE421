import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import time
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import sys

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
    # print(tlogs.min())

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
    run_part_1 = False
    run_part_2 = True

    if run_part_1:
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
        try:
            hidden_layer_size = int(sys.argv[1])
        except Exception:
            hidden_layer_size = 1000
            print('hidden_layer_size =', hidden_layer_size)
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

        gamma = 0.99
        alpha = 0.005

        #%% Training
        n_epoches = 200
        results = []

        fig, ax = plt.subplots(2, 1, sharex='col', figsize=(8, 6))
        plt.ion()

        tt = time.perf_counter()
        for i in range(n_epoches):
            # Forward pass
            Shidden, Xhidden, Sout, Xout, pred = forward(trainData, Whidden, bhidden, Wout, bout)
            assert pred.shape == trainTarget.shape
            accuracy = np.count_nonzero(pred == trainTarget) * 1.0 / trainTarget.size

            ce = CE(newtrain, Xout)

            _, _, _, validxout, validpred = forward(validData, Whidden, bhidden, Wout, bout)
            validaccuracy = np.count_nonzero(validpred == validTarget) * 1.0 / validTarget.size
            validce = CE(newvalid, validxout)

            _, _, _, testxout, testpred = forward(testData, Whidden, bhidden, Wout, bout)
            testaccuracy = np.count_nonzero(testpred == testTarget) * 1.0 / testTarget.size
            testce = CE(newvalid, validxout)

            results.append((ce, validce, testce, accuracy, validaccuracy, testaccuracy))

            if i % 3 == 0 or i == n_epoches - 1:
                elapsed = time.perf_counter() - tt
                per = elapsed / (i + 1)
                print('\r', i, '/', ','.join(['%.2f' % i for i in results[-1]]), n_epoches, '%d' % elapsed, '+', '%d' % (per * (n_epoches - i)), '(', per, ')', end='    \r')
                ax[0].clear()
                ax[0].plot(np.asarray(results)[:, 0:3])
                ax[0].set_ylabel('Average Cross Entropy Loss')
                ax[0].set_xlabel('Epoches')
                ax[0].set_xlim([0, n_epoches])
                ax[0].set_title('Hidden Layer Size = %d' % hidden_layer_size)
                ax[0].legend(['Training', 'Validation', 'Testing'])
                ax[0].grid(True)
                ax[1].clear()
                ax[1].plot(np.asarray(results)[:, 3:6])
                ax[1].set_ylabel('Accuracy')
                ax[1].set_xlabel('Epoches')
                ax[1].set_ylim([0, 1])
                ax[1].set_xlim([0, n_epoches])
                ax[1].grid(True)

                # fig.show()
            fig.savefig('%d_%g_%g.tmp.png' % (hidden_layer_size, gamma, alpha), dpi=75)

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

            Whidden_v_new = gamma * Whidden_v_old + alpha * dl_dWhidden
            bhidden_v_new = gamma * bhidden_v_old + alpha * dl_dbhidden
            Wout_v_new = gamma * Wout_v_old + alpha * dl_dWout
            bout_v_new = gamma * bout_v_old + alpha * dl_dbout

            Whidden -= Whidden_v_new
            bhidden -= bhidden_v_new
            Wout -= Wout_v_new
            bout -= bout_v_new

            assert np.all(np.isfinite(Whidden))
            assert np.all(np.isfinite(bhidden))
            assert np.all(np.isfinite(Wout))
            assert np.all(np.isfinite(bout))

            Whidden_v_old = Whidden_v_new
            bhidden_v_old = bhidden_v_new
            Wout_v_old = Wout_v_new
            bout_v_old = bout_v_new

        fig.savefig('%d_%g_%g.png' % (hidden_layer_size, gamma, alpha), dpi=300)
        fig.savefig('%d_%g_%g.pdf' % (hidden_layer_size, gamma, alpha))
        np.savetxt('%d_%g_%g.csv' % (hidden_layer_size, gamma, alpha), results, fmt='%.12g', delimiter=',')

    #%% neural network training
    if run_part_2:

        # %% Initialize dataset
        trainData, validData, testData, trainTarget, validTarget, testTarget = loadData()
        trainData = trainData[..., np.newaxis]
        validData = validData[..., np.newaxis]
        testData = testData[..., np.newaxis]
        newtrain, newvalid, newtest = convertOneHot(trainTarget, validTarget, testTarget)
        newtrain = newtrain
        newvalid = newvalid
        newtest = newtest

        n_class = 10
        batch_size = 32
        n_epoches = 50

        # %% Build model
        # Model parameters (Variables)
        # filter_height, filter_width, in_channels, out_channels
        # The Glorot normal initializer, also called Xavier normal initializer.
        rand = np.random.RandomState(421)
        tf.set_random_seed(421)
        w0 = tf.get_variable('w0', shape=(3, 3, 1, 32), initializer=tf.contrib.layers.xavier_initializer(uniform=False))
        w1 = tf.get_variable('w1', shape=(14 * 14 * 32, 784), initializer=tf.contrib.layers.xavier_initializer(uniform=False))
        w2 = tf.get_variable('w2', shape=(784, 10), initializer=tf.contrib.layers.xavier_initializer(uniform=False))

        b0 = tf.get_variable('B0', shape=(32, ), initializer=tf.initializers.zeros())
        b1 = tf.get_variable('B1', shape=(784, ), initializer=tf.initializers.zeros())
        b2 = tf.get_variable('B2', shape=(10, ), initializer=tf.initializers.zeros())

        # Data inputs (Placeholders)
        x = tf.placeholder(
            dtype=tf.float32,
            shape=[None, 28, 28, 1],  # batch, in_height, in_width, in_channels
            name='x'
        )
        y = tf.placeholder(
            dtype=tf.float32,
            shape=[None, n_class],
            name='y'
        )
        lambd = tf.placeholder(
            dtype=tf.float32,
            shape=(1, 1),
            name='lambda'
        )
        dropout = tf.placeholder(
            dtype=tf.float32,
            shape=(),
            name='dropout'
        )

        # 2. Convolution Layer
        # Input N x 28 x 28 x 1
        # Output N x 26 x 26 x 32
        # [1, stride, stride, 1]
        conv = tf.nn.conv2d(x, w0, strides=[1, 1, 1, 1], padding='SAME')
        conv = tf.nn.bias_add(conv, b0)

        # 3. ReLU activation
        conv = tf.nn.relu(conv)

        # 4. batch normalization
        batch_norm = tf.layers.batch_normalization(conv)

        # 5. Max Pooling
        # Output N x 13 x 13 x 32
        max_pool = tf.layers.max_pooling2d(batch_norm, pool_size=2, strides=2, padding='SAME')

        # fc1 = tf.reshape(max_pool, [-1, w1.get_shape().as_list()[0]])
        # 6. Flatten layer
        # Output N x (13 x 13 x 32)
        max_pool = tf.layers.flatten(max_pool)

        # 7. Fully connected layer
        fc1 = tf.matmul(max_pool, w1) + b1

        # Drop out after 7
        fc1 = tf.nn.dropout(fc1, 1 - dropout)

        # 8. ReLU activation
        fc1 = tf.nn.relu(fc1)
        # Output N x 784

        # 9. Fully connected layer
        out = tf.matmul(fc1, w2) + b2
        # Output N x 10

        # 10. Softmax output
        prob = tf.nn.softmax(out)  # softmax probability
        yhat = tf.argmax(out, axis=-1)

        # 11. Cross Entropy Loss
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(labels=y, logits=prob)

        regularizer = tf.contrib.layers.l2_regularizer(scale=lambd)
        loss = tf.reduce_mean(cross_entropy)
        loss += regularizer(w0) + regularizer(w1) + regularizer(w2)
        optimizer = tf.train.AdamOptimizer(learning_rate=1.0e-4)

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

        train_op = optimizer.minimize(loss)
        train_op = tf.group([train_op, update_ops])

        params = [
            (0, 0),
            (0.01, 0),
            (0.1, 0),
            (0.5, 0),
            (0, 0.9),
            (0, 0.75),
            (0, 0.5),
        ]
        params = np.asarray(params)
        results_all = [[] for _ in range(len(params))]
        for (lambd_val, dropprob_val), results in zip(params, results_all):

            # Dictionary to feed in for reporting
            train_dict = {x: trainData,
                          y: newtrain,
                          lambd: [[lambd_val]],
                          dropout: dropprob_val,}
            valid_dict = {x: validData,
                          y: newvalid,
                          lambd: [[lambd_val]],
                          dropout: dropprob_val,}
            test_dict = {x: testData,
                         y: newtest,
                         lambd: [[lambd_val]],
                         dropout: dropprob_val,}

            init = tf.global_variables_initializer()
            with tf.Session() as sess:
                sess.run(init)

                print('iter', 'train_loss', 'valid_loss', 'test_loss', 'train_acc', 'valid_acc', 'test_acc')
                for i in range(n_epoches):
                    # Record losses and accuracy
                    valid_loss, valid_yhat = sess.run([
                            loss, yhat
                            ], feed_dict=valid_dict)
                    valid_loss = np.asscalar(valid_loss)
                    valid_acc = np.count_nonzero(valid_yhat == validTarget) / validTarget.size
                    
                    test_loss, test_yhat = sess.run([
                            loss, yhat
                            ], feed_dict=test_dict)
                    test_loss = np.asscalar(test_loss)
                    test_acc = np.count_nonzero(test_yhat == testTarget) / testTarget.size
                    
                    train_loss, train_yhat = sess.run([
                            loss, yhat
                            ], feed_dict=train_dict)
                    train_loss = np.asscalar(train_loss)
                    train_acc = np.count_nonzero(train_yhat == trainTarget) / trainTarget.size

                    results.append(
                                (train_loss, valid_loss, test_loss, train_acc, valid_acc, test_acc)
                                )
                    print(i, results[-1])

                    # Minibatch
                    perm = rand.permutation(trainData.shape[0])
                    for start in range(0, trainData.shape[0], batch_size):
                        chunk = perm[start:start+batch_size]
                        sess.run(train_op, feed_dict={x: trainData[chunk, ...],
                                                      y: newtrain[chunk, ...],
                                                      lambd: [[lambd_val]],
                                                      dropout: dropprob_val,})

        results_all = np.asarray(results_all)
        for i, res in enumerate(results_all):
            np.savetxt('part2_%g_%g.csv' % tuple(params[i]), res, fmt='%.12g', delimiter=',')

        fig, ax = plt.subplots(2, 3, sharey='row', sharex='all', figsize=(8, 6))
        for i in range(3):
            ax[0, i].plot(results_all[0:4, :, i].T)
            ax[0, i].set_ylabel('Average Cross Entropy Loss')
            ax[0, i].set_xlabel('Epoches')
            ax[0, i].set_xlim([0, n_epoches])
            ax[0, i].grid(True)
        for i in range(3):
            ax[1, i].plot(results_all[0:4, :, i+3].T)
            ax[1, i].set_ylabel('Accuracy')
            ax[1, i].set_xlabel('Epoches')
            ax[1, i].set_ylim([0, 1])
            ax[1, i].set_xlim([0, n_epoches])
            ax[1, i].grid(True)
        ax[0, 0].legend([r'$\lambda=%g, p=%g$' % tuple(i) for i in params[0:4]])
        ax[0, 0].set_title('Training')
        ax[0, 1].set_title('Validation')
        ax[0, 2].set_title('Testing')
        fig.savefig('part2_1.png', dpi=300)
        fig.savefig('part2_1.pdf')

        fig, ax = plt.subplots(2, 3, sharey='row', sharex='all', figsize=(8, 6))
        for i in range(3):
            ax[0, i].plot(results_all[[0, 4, 5, 6], :, i].T)
            ax[0, i].set_ylabel('Average Cross Entropy Loss')
            ax[0, i].set_xlabel('Epoches')
            ax[0, i].set_xlim([0, n_epoches])
            ax[0, i].grid(True)
        for i in range(3):
            ax[1, i].plot(results_all[[0, 4, 5, 6], :, i+3].T)
            ax[1, i].set_ylabel('Accuracy')
            ax[1, i].set_xlabel('Epoches')
            ax[1, i].set_ylim([0, 1])
            ax[1, i].set_xlim([0, n_epoches])
            ax[1, i].grid(True)
        ax[0, 0].legend([r'$\lambda=%g, p=%g$' % tuple(i) for i in params[[0, 4, 5, 6]]])
        ax[0, 0].set_title('Training')
        ax[0, 1].set_title('Validation')
        ax[0, 2].set_title('Testing')
        fig.savefig('part2_2.png', dpi=300)
        fig.savefig('part2_2.pdf')
        

