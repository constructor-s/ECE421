import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

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
    W = np.concatenate([b, W])
    x = np.concatenate([-np.ones([1, x.shape[1]]), x], axis=0)
    
    err = W.T.dot(x) - y
    mse = err.dot(err.T) * 1.0 / 2.0 / x.shape[1]
    regularization = reg * 0.5  * W.T.dot(W)

    return mse + regularization    
    

def gradMSE(W, b, x, y, reg):
    # Your implementation here
    W = np.concatenate([b, W])
    x = np.concatenate([-np.ones([1, x.shape[1]]), x], axis=0)
    
    err = W.T.dot(x) - y
    mse = (err * 1.0 / x.shape[1]).dot(x.T).T
    regularization = reg * W
    
    grad = mse + regularization
    
    return grad[1:], grad[0]

def crossEntropyLoss(W, b, x, y, reg):
    # Your implementation here
    pass

def gradCE(W, b, x, y, reg):
    # Your implementation here
    pass

def grad_descent(W, b, trainingData, trainingLabels, alpha, iterations, reg, EPS):
    # Your implementation here
    mseList = []
    for i in range(iterations):
        print('Iteration', i, end='    \r')
        gradW, gradB= gradMSE(W, b, trainingData, trainingLabels, reg)
        W = W - alpha * gradW
        b = b - alpha * gradB
        mse = MSE(W, b, trainingData, trainingLabels, reg)
        mse = np.asscalar(mse)
        mseList.append(mse)
        if np.abs(mse) < EPS:
            break
        
    return W, b, mseList

def buildGraph(beta1=None, beta2=None, epsilon=None, lossType=None, learning_rate=None):
    # Your implementation here
    pass


if __name__ == '__main__':
    trainData, validData, testData, trainTarget, validTarget, testTarget = loadData()
    
    trainDataVec = trainData.reshape([trainData.shape[0], trainData.shape[1]*trainData.shape[2]]).T
    validDataVec = validData.reshape([validData.shape[0], validData.shape[1]*validData.shape[2]]).T
    testDataVec = testData.reshape([testData.shape[0], testData.shape[1]*testData.shape[2]]).T
    
    for alpha in [0.005, 0.001, 0.0001]:
        
        W = np.zeros([trainDataVec.shape[0], 1])
        b = np.array([[0]])
        W, b, mseList = grad_descent(W, b, trainDataVec, trainTarget.T, alpha, 500, 0, 1e-6)
        
        
        break
    
    
    
    
    