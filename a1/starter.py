import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

import time
from scipy.special import expit

def loadData():
    with np.load('notMNIST.npz') as data :
        Data, Target = data ['images'], data['labels']
        posClass = 2
        negClass = 9
        dataIndx = (Target==posClass) + (Target==negClass)
        Data = Data[dataIndx]/255.
        Target = Target[dataIndx].reshape(-1, 1)
        Target[Target==posClass] = 1
        Target[Target==negClass] = 0
        np.random.seed(421)
        randIndx = np.arange(len(Data))
        np.random.shuffle(randIndx)
        Data, Target = Data[randIndx], Target[randIndx]
        trainData, trainTarget = Data[:3500], Target[:3500]
        validData, validTarget = Data[3500:3600], Target[3500:3600]
        testData, testTarget = Data[3600:], Target[3600:]
    return trainData, validData, testData, trainTarget, validTarget, testTarget

def MSE(W, b, x, y, reg):
    # Your implementation here
#    W = np.concatenate([b, W])
#    x = np.concatenate([-np.ones([1, x.shape[1]]), x], axis=0)
    
    err = W.T.dot(x) + b - y
    mse = err.dot(err.T) * 1.0 / 2.0 / x.shape[1]
    regularization = reg * 0.5  * W.T.dot(W)

    return mse + regularization    

def gradMSE(W, b, x, y, reg):
    # Your implementation here
#    W = np.concatenate([b, W])
#    x = np.concatenate([-np.ones([1, x.shape[1]]), x], axis=0)
    
    err = W.T.dot(x) + b - y
    mse = 1.0 / x.shape[1] * x.dot(err.T)
    regularization = reg * W
    
    grad = mse + regularization
    
    gradB = np.sum(1.0 / x.shape[1] * err)
    
    return grad, gradB

def crossEntropyLoss(W, b, x, y, reg):
    # Your implementation here
    z = W.T.dot(x) + b
    yhat = expit(z)
    
    N = y.size
    
#    cel = 1.0 / N * np.sum(
#            -y * np.log(yhat)
#            -(1-y) * np.log(1-yhat)
#            ) + reg / 2.0 * W.T.dot(W)
    
    assert np.all((y == 0) | (y == 1))
    sum_parts = np.zeros_like(yhat)
    sum_parts[y==0] = -np.log(1-yhat[y==0])
    sum_parts[y==1] = -np.log(yhat[y==1])
    cel = 1.0 / N * np.sum(sum_parts) + reg / 2.0 * W.T.dot(W)

    # cel = 1.0 / N * np.sum(
    #         -y * x -
    #         -np.log(1-yhat)
    #         ) + reg / 2.0 * W.T.dot(W)
    
    return cel

def gradCE(W, b, x, y, reg):
    """

    :param W: 784x1
    :param b: 1x1
    :param x: 784xn
    :param y: 1xn
    :param reg: scalar
    :type reg: float
    :return:
    """
    # Your implementation here
    z = W.T.dot(x) + b  # 1xn
    yhat = expit(z)  # 1xn

    # dyhatdb = yhat * (1-yhat)  # 1xn
    # dyhatdw = dyhatdb * x  # 1xn

    N = y.size

    dldw = 1.0 / N * x.dot((yhat - y).T) + reg * W  # 784x1

    dldb = 1.0 / N * np.sum(
        yhat - y
    )  # float

    return dldw, dldb

def grad_descent(W, b, trainingData, trainingLabels, alpha, iterations, reg, EPS, lossType=None,
                 validData=None, validTarget=None, testData=None, testTarget=None):
    # Your implementation here
    if lossType is None or lossType == 'mse':
        errorFun = MSE
        gradErrorFun = gradMSE
        accfun = lambda x, y, W, b: np.count_nonzero((W.T.dot(x) + b > 0.5) == y) * 1.0 / y.size
    elif lossType == 'ce':
        errorFun = crossEntropyLoss
        gradErrorFun = gradCE
        accfun = lambda x, y, W, b: np.count_nonzero((expit(W.T.dot(x) + b) > 0.5) == y) * 1.0 / y.size
    else:
        raise Exception('Invalid lossType = %s' % lossType)

    mseList = []
    validMseList = []
    testMseList = []
    accuracyList = []
    validAccuracyList = []
    testAccuracyList = []
    for i in range(iterations):
        if i % 50 == 49:
            print('\rIteration %03d/%03d' % (i, iterations), end='\r')
        
        mse = errorFun(W, b, trainingData, trainingLabels, reg)
        mse = np.asscalar(mse)
        mseList.append(mse)
        
        if validData is not None and validTarget is not None:
            validMseList.append(np.asscalar(
                    errorFun(W, b, validData, validTarget, reg)
                    ))
        if testData is not None and testTarget is not None:
            testMseList.append(np.asscalar(
                    errorFun(W, b, testData, testTarget, reg)
                    ))

        accuracy = lambda x, y: accfun(x, y, W, b)
#        print(np.min(expit(W.T.dot(trainDataVec) + b)), np.max(expit(W.T.dot(trainDataVec) + b)))
        accuracyList.append(accuracy(trainDataVec, trainingLabels))
        validAccuracyList.append(accuracy(validDataVec, validTarget))
        testAccuracyList.append(accuracy(testDataVec, testTarget))
        
        if np.abs(mse) < EPS:
            break
        
        gradW, gradB = gradErrorFun(W, b, trainingData, trainingLabels, reg)
        W = W - alpha * gradW
        b = b - alpha * gradB
        
    print()    
    if validData is None or validTarget is None or testData is None or testTarget is None:
        return W, b, mseList
    else:
        return W, b, mseList, validMseList, testMseList, accuracyList, validAccuracyList, testAccuracyList

def buildGraph(beta1=0.9, beta2=0.999, epsilon=1e-08, 
               lossType=None, learning_rate=0.001, d=784, stddev=0.5):
    # Validate input
    if lossType is None or lossType == 'mse':
        pass
    elif lossType == 'ce':
        pass
    else:
        raise Exception('Invalid lossType = %s' % lossType)
    
    tf.set_random_seed(421)
    
    # Model parameters (Variables)
    W = tf.Variable(tf.truncated_normal(
        (d, 1), stddev=stddev, seed=421
    ), trainable=True, name='Weight')
#    W = tf.Variable(tf.zeros(
#        (d, 1)
#    ), trainable=True, name='Weight')
    b = tf.Variable(tf.truncated_normal(
        (1, 1), stddev=stddev, seed=421
    ), trainable=True, name='bias')
    
    # Data inputs (Placeholders)
    x = tf.placeholder(
        dtype=tf.float32,
        shape=(d, None),
        name='data'
    )
    y = tf.placeholder(
        dtype=tf.float32,
        shape=(1, None),
        name='label'
    )
    
    # Hyperparameter (Placeholder)
    lambd = tf.placeholder(
        dtype=tf.float32,
        shape=(1, 1),
        name='lambda'
    )
    
    if lossType is None or lossType == 'mse':
        yhat = tf.add(tf.matmul(tf.transpose(W), x), b)
        loss = tf.losses.mean_squared_error(y, yhat)
    else:
        yhat = tf.sigmoid(tf.add(tf.matmul(tf.transpose(W), x), b))
        loss = tf.losses.log_loss(y, yhat)
        
#    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(loss)
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate,
                                        beta1=beta1,
                                        beta2=beta2,
                                        epsilon=epsilon,
                                    ).minimize(loss)
    
    return optimizer, W, b, x, y, lambd, yhat, loss



if __name__ == '__main__':
    # %%
    # Q1
    linear_regression_epochs = 0
    run13 = True
    run14 = True
    save_linear_regression = False
    plot_linear_regression = True
    
    # Q2
    logistic_regression_epochs = 0
    
    # Q3
    tf_epochs = 700
    run33 = False
    run34 = True
    
    trainData, validData, testData, trainTarget, validTarget, testTarget = loadData()
    
    trainDataVec = trainData.reshape([trainData.shape[0], trainData.shape[1]*trainData.shape[2]]).T
    validDataVec = validData.reshape([validData.shape[0], validData.shape[1]*validData.shape[2]]).T
    testDataVec = testData.reshape([testData.shape[0], testData.shape[1]*testData.shape[2]]).T


    def train(alphaLambdList, figfilename=None, epochs=linear_regression_epochs, lossType=None):
        mse = []
        validMse = []
        testMse = []
        weight = []
        bias = []
        accu = []
        validAccu = []
        testAccu = []
        runtime = []

        for alpha, lambd in alphaLambdList:
            print('Alpha =', alpha, 'Lambda =', lambd)

            W = np.zeros([trainDataVec.shape[0], 1])
#            W = np.random.RandomState(42).rand(trainDataVec.shape[0], 1)
            b = np.array([[0]])
            
            tic = time.clock()
            W, b, mseList, validMseList, testMseList, accuracyList, validAccuracyList, testAccuracyList = grad_descent(
                W, b, trainDataVec, trainTarget.T, alpha,
                epochs, lambd, 0, lossType,
                validDataVec, validTarget.T, testDataVec, testTarget.T)
            runtime.append(time.clock() - tic)

            mse.append(mseList)
            validMse.append(validMseList)
            testMse.append(testMseList)

            weight.append(W)
            bias.append(b)

            accu.append(accuracyList)
            validAccu.append(validAccuracyList)
            testAccu.append(testAccuracyList)

        mse = np.array(mse)
        validMse = np.array(validMse)
        testMse = np.array(testMse)

        weight = np.array(weight)
        bias = np.array(bias)

        accu = np.array(accu)
        validAccu = np.array(validAccu)
        testAccu = np.array(testAccu)

        if save_linear_regression and figfilename is not None:
            np.savez(figfilename + '.npz', mse=mse, weight=weight, bias=bias)

        if plot_linear_regression:
            fig, axs = plt.subplots(2, 3, figsize=(15, 10), sharey='row')
            for i, (ax, err, title) in enumerate(zip(
                    axs.ravel(),
                    (mse, validMse, testMse, accu, validAccu, testAccu),
                    ('Training Loss', 'Validation Loss', 'Testing Loss',
                     'Training Accuracy', 'Validation Accuracy', 'Testing Accuracy'),
            )):
                ax.plot(err.T)
                ax.set_title(title)
                ax.grid()
                ax.set_xlabel('Epoches')
                if i <= 2:
                    if lossType is None or lossType == 'mse':
                        ax.set_ylabel('Mean Square Error')
                    elif lossType == 'ce':
                        ax.set_ylabel('Cross Entropy Loss')
                    else:
                        raise Exception('Invalid lossType = %s' % lossType)
                else:
                    ax.set_ylabel('Accuracy')

            axs[0, 0].legend([r'$\alpha=%g,\lambda=%g$' % (alpha, lambd)
                            for alpha, lambd in alphaLambdList])
            if figfilename:
                plt.savefig(figfilename, dpi=150)
            
            print('Alpha', 'Lambda', 'Runtime',
                  'Training Loss', 'Validation Loss', 'Testing Loss',
                  'Training Accuracy', 'Validation Accuracy', 'Testing Accuracy', sep='\t')
            for (alpha, lambd), mseList, validList, testList, accuList, validAccuList, testAccuList, t in zip(
                    alphaLambdList, mse, validMse, testMse, accu, validAccu, testAccu, runtime):
                print(alpha, lambd, t,
                      mseList[-1], validList[-1], testList[-1],
                      accuList[-1], validAccuList[-1], testAccuList[-1],
                      sep='\t')

    # %% Q1
    if linear_regression_epochs:
        #%% Part 1.3
        if run13:
            train([(0.005, 0), (0.001, 0), (0.0001, 0)], figfilename='fig13.png')
            
        #%% Part 1.4
        if run14:
            train([(0.005, 0.001), (0.005, 0.1), (0.005, 0.5)], figfilename='fig14.png')
            
        #%% 1.5 Normal equation
        print('1.5 Normal equation')
        X = trainDataVec.T
        X = np.hstack([np.ones([X.shape[0], 1]), X])
        tic = time.clock()
        W = np.linalg.pinv(X).dot(trainTarget)
        print('Runtime', time.clock() - tic)
        print('Training Accuracy\tValidation Accuracy\tTesting Accuracy')
        
        def accuracy(x, y):
            X = x.T
            X = np.hstack([np.ones([X.shape[0], 1]), X])            
            return np.count_nonzero(((X.dot(W)) > 0.5) == y) * 1.0 / len(y)
        
        print('%.3f' % accuracy(trainDataVec, trainTarget), '\t', 
              '%.3f' % accuracy(validDataVec, validTarget), '\t', 
              '%.3f' % accuracy(testDataVec, testTarget))

    # %% Q2
    if logistic_regression_epochs:
        train([(0.005, 0.001), (0.005, 0.1), (0.005, 0.5)], figfilename='fig23.png',
              epochs=logistic_regression_epochs, lossType='ce')
        
        
    # %% Q3
    if tf_epochs:
        def tf_train(beta1=0.9, beta2=0.999, epsilon=1e-08, 
               lossType=None, learning_rate=0.001, stddev=0.5, 
               batch_size=500, lambd_val=0, tf_epochs=tf_epochs):
            
            print('beta1', 'beta2', 'epsilon', 'lossType', 'learning_rate', 'stddev', 'batch_size', 'lambd_val', 'tf_epochs', sep='\t')
            print(beta1, beta2, epsilon, lossType, learning_rate, stddev, batch_size, lambd_val, tf_epochs, sep='\t')
            
            optimizer, W, b, x, y, lambd, yhat, loss = buildGraph(beta1=beta1, beta2=beta2, epsilon=epsilon, 
               lossType=lossType, learning_rate=learning_rate, stddev=stddev)
            
            train_dict = {x: trainDataVec,
                          y: trainTarget.T,
                          lambd: ((lambd_val,),)}
            valid_dict = {x: validDataVec,
                          y: validTarget.T,
                          lambd: ((lambd_val,),)}
            test_dict = {x: testDataVec,
                         y: testTarget.T,
                         lambd: ((lambd_val,),)}
            
            results = []
            
            rand = np.random.RandomState(421)
            init = tf.global_variables_initializer()
            with tf.Session() as sess:
                sess.run(init)
                
                print('iter', 'train_loss', 'train_acc', sep='\t')
                for i in range(tf_epochs):
                    valid_loss, valid_yhat = sess.run([
                            loss, yhat
                            ], feed_dict=valid_dict)
                    valid_acc = np.count_nonzero((valid_yhat > 0.5) == validTarget.T) / validTarget.size
                    
                    test_loss, test_yhat = sess.run([
                            loss, yhat
                            ], feed_dict=test_dict)
                    test_acc = np.count_nonzero((test_yhat > 0.5) == testTarget.T) / testTarget.size
                    
                    train_loss, train_yhat = sess.run([
                            loss, yhat
                            ], feed_dict=train_dict)
                    train_acc = np.count_nonzero((train_yhat > 0.5) == trainTarget.T) / trainTarget.size
                    
                    results.append(
                            (train_loss, valid_loss, test_loss, train_acc, valid_acc, test_acc)
                            )
                    
                    if i % 100 == 0 or i == tf_epochs-1:
                        print('\r%d' % i, '%.3f' % train_loss, '%.3f' % train_acc, sep='\t', end='    \r')
                    
                    # Minibatch
                    perm = rand.permutation(trainDataVec.shape[1])
                    for start in range(0, trainDataVec.shape[1], batch_size):
                        chunk = (slice(None, None), perm[start:start+batch_size])
                        sess.run(optimizer, feed_dict={x: trainDataVec[chunk],
                                                       y: trainTarget.T[chunk],
                                                       lambd: ((lambd_val,),)})
                    
            print()                    
            return np.array(results)

        #%%
        if run33:
            res = []
            lossType = 'mse'
            bsizes = (100, 500, 700, 1750)
            fig, axs = plt.subplots(2, 3, figsize=(15, 10), sharey='row')
            for bsize in bsizes:
                tic = time.clock()
                curves = tf_train(lossType=lossType, batch_size=bsize)
                print('Runtime:', time.clock() - tic)

                for i, (line, ax, title) in enumerate(zip(curves.T, axs.ravel(),
                                           ('Training Loss', 'Validation Loss', 'Testing Loss',
                                            'Training Accuracy', 'Validation Accuracy', 'Testing Accuracy'))):
                    ax.plot(line)
                    ax.set_title(title)
                    ax.set_xlabel('Epoches')
                    if i <= 2:
                        if lossType is None or lossType == 'mse':
                            ax.set_ylabel('Mean Square Error')
                        elif lossType == 'ce':
                            ax.set_ylabel('Cross Entropy Loss')
                        else:
                            raise Exception('Invalid lossType = %s' % lossType)
                    else:
                        ax.set_ylabel('Accuracy')
                    ax.grid(True)

            axs[0, 0].legend(['B=%d' % i for i in bsizes])
            fig.savefig('fig33.png', dpi=150)

        #%%
        if run34:
            parameters = ((0.9, 0.999, 1e-08,   'Default(0.9,0.999,1e-8)'),
                          (0.95, 0.999, 1e-08,  r'$\beta_1=0.95$'),
                          (0.99, 0.999, 1e-08,  r'$\beta_1=0.99$'),
                          (0.9, 0.99, 1e-08,    r'$\beta_2=0.99$'),
                          (0.9, 0.9999, 1e-08,  r'$\beta_2=0.9999$'),
                          (0.9, 0.999, 1e-09,   r'$\epsilon=1e-9$'),
                          (0.9, 0.999, 1e-04,   r'$\epsilon=1e-4$'),
                          )

            res = []
            legendList = []
            lossType = 'mse'
            for j, (beta1, beta2, eps, name) in enumerate(parameters):
                tic = time.clock()
                curves = tf_train(lossType=lossType, batch_size=500, beta1=beta1, beta2=beta2, epsilon=eps)
                print('Runtime:', time.clock() - tic)

                if j in (0, 3, 5):
                    if j == 0:
                        baseline = curves
                        fig, axs = plt.subplots(2, 3, figsize=(15, 10), sharey='row')
                    else:
                        axs[0, 0].legend(legendList)
                        fig.savefig('fig34%d.png' % j, dpi=150)
                        fig.savefig('fig34%d.pdf' % j)

                        fig, axs = plt.subplots(2, 3, figsize=(15, 10), sharey='row')
                        legendList = [parameters[0][-1]]
                        for i, (line, ax, title) in enumerate(zip(baseline.T, axs.ravel(),
                                                                  ('Training Loss', 'Validation Loss', 'Testing Loss',
                                                                   'Training Accuracy', 'Validation Accuracy', 'Testing Accuracy'))):
                            ax.plot(line)
                            ax.set_title(title)
                            ax.set_xlabel('Epoches')
                            if i <= 2:
                                if lossType is None or lossType == 'mse':
                                    ax.set_ylabel('Mean Square Error')
                                elif lossType == 'ce':
                                    ax.set_ylabel('Cross Entropy Loss')
                                else:
                                    raise Exception('Invalid lossType = %s' % lossType)
                            else:
                                ax.set_ylabel('Accuracy')
                            ax.grid(True)

                legendList.append(name)
                for i, (line, ax, title) in enumerate(zip(curves.T, axs.ravel(),
                                                          ('Training Loss', 'Validation Loss', 'Testing Loss',
                                                           'Training Accuracy', 'Validation Accuracy', 'Testing Accuracy'))):
                    ax.plot(line)
                    ax.set_title(title)
                    ax.set_xlabel('Epoches')
                    if i <= 2:
                        if lossType is None or lossType == 'mse':
                            ax.set_ylabel('Mean Square Error')
                        elif lossType == 'ce':
                            ax.set_ylabel('Cross Entropy Loss')
                        else:
                            raise Exception('Invalid lossType = %s' % lossType)
                    else:
                        ax.set_ylabel('Accuracy')
                    ax.grid(True)

            axs[0, 0].legend(legendList)
            fig.savefig('fig34%d.png' % j, dpi=150)
            fig.savefig('fig34%d.pdf' % j)
