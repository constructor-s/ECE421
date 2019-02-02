# import tensorflow as tf
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
    
    cel = 1.0 / N * np.sum(
            -y * np.log(yhat) - 
            -(1-y) * np.log(1-yhat)            
            ) + reg / 2.0 * W.T.dot(W)
    
    return cel

def gradCE(W, b, x, y, reg):
    # Your implementation here
    z = W.T.dot(x) + b
    yhat = expit(z)

    dyhatdb = yhat * (1-yhat)
    dyhatdw = dyhatdb * x

    N = y.size

    dldw = 1.0 / N * np.sum(
        -y * 1.0 / yhat * dyhatdw +
        (1-y) * 1.0 / (1-yhat) * dyhatdw
    ) + reg * W

    dldb = 1.0 / N * np.sum(
        -y * 1.0 / yhat * dyhatdb +
        (1-y) * 1.0 / (1-yhat) * dyhatdb
    )

    return dldw, dldb

def grad_descent(W, b, trainingData, trainingLabels, alpha, iterations, reg, EPS, lossType=None,
                 validData=None, validTarget=None, testData=None, testTarget=None):
    # Your implementation here
    if lossType is None or lossType == 'mse':
        errorFun = MSE
        gradErrorFun = gradMSE
    else:
        errorFun = crossEntropyLoss
        gradErrorFun = gradCE

    mseList = []
    validMseList = []
    testMseList = []
    for i in range(iterations):
        if i % 50 == 49:
            print('\rIteration %03d/%03d' % (i, iterations), end='\r')
        gradW, gradB = gradErrorFun(W, b, trainingData, trainingLabels, reg)
        W = W - alpha * gradW
        b = b - alpha * gradB
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
        
        if np.abs(mse) < EPS:
            break
        
    print()    
    if validData is None or validTarget is None or testData is None or testTarget is None:
        return W, b, mseList
    else:
        return W, b, mseList, validMseList, testMseList

def buildGraph(beta1=None, beta2=None, epsilon=None, lossType=None, learning_rate=None):
    # Your implementation here
    pass


if __name__ == '__main__':
    # %%
    linear_regression_epochs = 5000
    run13 = True
    run14 = True
    save_linear_regression = False
    plot_linear_regression = True

    logistic_regression_epochs = 5000
    
    run_sgd = True    
    
    trainData, validData, testData, trainTarget, validTarget, testTarget = loadData()
    
    trainDataVec = trainData.reshape([trainData.shape[0], trainData.shape[1]*trainData.shape[2]]).T
    validDataVec = validData.reshape([validData.shape[0], validData.shape[1]*validData.shape[2]]).T
    testDataVec = testData.reshape([testData.shape[0], testData.shape[1]*testData.shape[2]]).T


    def train(alphaLambdList, figfilename=None, lossType=None):
        mse = []
        validMse = []
        testMse = []
        weight = []
        bias = []
        accu = []
        runtime = []

        for alpha, lambd in alphaLambdList:
            print('Alpha =', alpha, 'Lambda =', lambd)

            W = np.zeros([trainDataVec.shape[0], 1])
            b = np.array([[0]])
            tic = time.clock()
            W, b, mseList, validMseList, testMseList = grad_descent(
                W, b, trainDataVec, trainTarget.T, alpha,
                linear_regression_epochs, lambd, 1e-6, lossType,
                validDataVec, validTarget.T, testDataVec, testTarget.T)
            runtime.append(time.clock() - tic)

            mse.append(mseList)
            validMse.append(validMseList)
            testMse.append(testMseList)

            weight.append(W)
            bias.append(b)

            def accuracy(x, y):
                y = y.T
                return np.count_nonzero((W.T.dot(x) + b > 0.5) == y) * 1.0 / y.size

            accu.append((
                accuracy(trainDataVec, trainTarget),
                accuracy(validDataVec, validTarget),
                accuracy(testDataVec, testTarget),
            ))

        mse = np.array(mse)
        validMse = np.array(validMse)
        testMse = np.array(testMse)

        weight = np.array(weight)
        bias = np.array(bias)

        accu = np.array(accu)

        if save_linear_regression and figfilename is not None:
            np.savez(figfilename + '.npz', mse=mse, weight=weight, bias=bias)

        if plot_linear_regression:
            fig, axs = plt.subplots(1, 3, figsize=(12, 4), sharey=True)
            for i, (err, title) in enumerate(zip(
                    (mse, validMse, testMse),
                    ('Training Loss', 'Validation Loss', 'Testing Loss'),
            )):
                axs[i].plot(err.T)
                axs[i].set_title(title)
                axs[i].grid()
                axs[i].set_xlabel('Epoches')
                axs[i].set_ylabel('Mean Square Error')

            axs[-1].legend([r'$\alpha=%g,\lambda=%g$' % (alpha, lambd)
                            for alpha, lambd in alphaLambdList])
            if figfilename:
                plt.savefig(figfilename, dpi=150)

            print('Alpha', 'Lambda', 'Runtime',
                  'Training Error', 'Validation Error', 'Testing Error',
                  'Training Accuracy', 'Validation Accuracy', 'Testing Accuracy', sep='\t')
            for (alpha, lambd), mseList, validList, testList, accuList, t in zip(
                    alphaLambdList, mse, validMse, testMse, accu, runtime):
                print(alpha, lambd, t,
                      mseList[-1], validList[-1], testList[-1],
                      *accuList,
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
        pass
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    