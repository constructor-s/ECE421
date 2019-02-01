# import tensorflow as tf
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
    pass

def gradCE(W, b, x, y, reg):
    # Your implementation here
    pass

def grad_descent(W, b, trainingData, trainingLabels, alpha, iterations, reg, EPS):
    # Your implementation here
    mseList = []
    for i in range(iterations):
        if i % 5 == 0:
            print('\rIteration %03d/%03d' % (i, iterations), end='\r')
        gradW, gradB= gradMSE(W, b, trainingData, trainingLabels, reg)
        W = W - alpha * gradW
        b = b - alpha * gradB
        mse = MSE(W, b, trainingData, trainingLabels, reg)
        mse = np.asscalar(mse)
        mseList.append(mse)
        if np.abs(mse) < EPS:
            break
        
    print()    
    return W, b, mseList

def buildGraph(beta1=None, beta2=None, epsilon=None, lossType=None, learning_rate=None):
    # Your implementation here
    pass


if __name__ == '__main__':
    linear_regression_epochs = 5000
    save_linear_regression = False
    plot_linear_regression = True
    run_logistic_regression = True
    run_sgd = True    
    
    trainData, validData, testData, trainTarget, validTarget, testTarget = loadData()
    
    trainDataVec = trainData.reshape([trainData.shape[0], trainData.shape[1]*trainData.shape[2]]).T
    validDataVec = validData.reshape([validData.shape[0], validData.shape[1]*validData.shape[2]]).T
    testDataVec = testData.reshape([testData.shape[0], testData.shape[1]*testData.shape[2]]).T
    
    if linear_regression_epochs:
        #%% Part 1.3
        mse = []
        weight = []
        bias = []
        alphaList = [0.005, 0.001, 0.0001]
        for alpha in alphaList:
            print('Alpha =', alpha)
            
            W = np.zeros([trainDataVec.shape[0], 1])
            b = np.array([[0]])
            W, b, mseList = grad_descent(W, b, trainDataVec, trainTarget.T, alpha, linear_regression_epochs, 0, 1e-6)
            
            mse.append(mseList)
            weight.append(W)
            bias.append(b)
            
        mse = np.array(mse)
        weight = np.array(weight)
        bias = np.array(bias)
        
        if save_linear_regression:
            np.savez('linear_regression_results13.npz', mse=mse, weight=weight, bias=bias)
        
        if plot_linear_regression:
            #%%
            plt.figure()
            plt.plot(mse.T)
            plt.legend([r'$\alpha=%g$' % i for i in alphaList])
            plt.grid()
            plt.xlabel('Epoches')
            plt.ylabel('Mean Square Error')
            plt.savefig('fig13.png', dpi=150)
            
            print('Alpha\tTraining Error')
            for alpha, mseList in zip(alphaList, mse):
                print(alpha, '\t', mseList[-1])
                
        #%% Part 1.4
        mse = []
        weight = []
        bias = []
        alpha = 0.005
        lambdList = (0.001, 0.1, 0.5)
        for lambd in lambdList:
            print('Lambda =', lambd)
            
            W = np.zeros([trainDataVec.shape[0], 1])
            b = np.array([[0]])
            W, b, mseList = grad_descent(W, b, trainDataVec, trainTarget.T, alpha, linear_regression_epochs, lambd, 1e-6)
            
            mse.append(mseList)
            weight.append(W)
            bias.append(b)
            
        mse = np.array(mse)
        weight = np.array(weight)
        bias = np.array(bias)
        
        if save_linear_regression:
            np.savez('linear_regression_results14.npz', mse=mse, weight=weight, bias=bias)
            
        if plot_linear_regression:
            #%%
            plt.figure()
            plt.plot(mse.T)
            plt.legend([r'$\lambda=%g$' % i for i in lambdList])
            plt.grid()
            plt.xlabel('Epoches')
            plt.ylabel('Mean Square Error')
            plt.savefig('fig14.png', dpi=150)
            
            print('Lambda\tTraining Error')
            for lambd, mseList in zip(lambdList, mse):
                print(lambd, '\t', mseList[-1])
            
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    