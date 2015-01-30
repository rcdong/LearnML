from __future__ import division

import numpy as np
import matplotlib.pyplot as plt
import time

def sigmoid(z):
    return 1. / (1. + np.exp(-z))

def trainLogRegres(train_x, train_y, opts):
    startTime = time.time()
    
    numSamples, numFeatures = train_x.shape
    alpha = opts['alpha']
    maxIter = opts['maxIter']
    optsType = opts['optimizeType']
    
    weights = np.ones((numFeatures, 1))
    
    if optsType == 1:
        for iter in range(maxIter):
            output = sigmoid(np.dot(train_x, weights))
            error = train_y - output
            weights = weights + alpha * np.dot(train_x.T, error)
    elif optsType == 2:
        for iter in range(maxIter):
            for i in xrange(numSamples):
                output = sigmoid(np.dot(train_x[i, :], weights))
                error = train_y[i, 0] - output
                weights = weights + alpha * error * train_x[i, :].reshape(numFeatures, 1)
    elif optsType == 3:
        for iter in range(maxIter):
            dataIndex = range(numSamples)
            for i in range(numSamples):
                alpha = 4. / (1. + iter + i) + 0.01
                randIndex = int(np.random.uniform(0, len(dataIndex)))
                output = sigmoid(np.dot(train_x[randIndex, :], weights))
                error = train_y[randIndex, 0] - output
                weights = weights + alpha * error * train_x[randIndex, :].reshape(numFeatures, 1)
                del(dataIndex[randIndex])
    else:
        raise NameError("optimize method type is not supported!")
    
    print "Congratulations, training complete! Took %f s" % (time.time() - startTime)    
    return weights
        
def testLogRegres(weights, test_x, test_y):
    numSamples= test_x.shape[0]
    matchCount = 0
    for i in xrange(numSamples):
        predict = sigmoid(np.dot(test_x[i, :], weights)) > 0.5
        if predict == bool(test_y[i, 0]):
            matchCount += 1
    accuracy = float(matchCount) / numSamples
    return accuracy

def showLogRegres(weights, train_x, train_y):
    numSamples = train_x.shape[0]
    for i in xrange(numSamples):
        if int(train_y[i, 0]) == 0:
            plt.plot(train_x[i, 1], train_x[i, 2], 'or')
        elif int(train_y[i, 0]) == 1:
            plt.plot(train_x[i, 1], train_x[i, 2], 'ob')
    
    min_x = min(train_x[:, 1])
    max_x = max(train_x[:, 1])
    y_min_x = (-weights[0,0] - weights[1,0] * min_x) / weights[2, 0]
    y_max_x = (-weights[0,0] - weights[1,0] * max_x) / weights[2, 0]
    plt.plot([min_x, max_x], [y_min_x, y_max_x], '-g')
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.show()
            

def loadData():
    train_x = []
    train_y = []
    fr = open("C:/Users/Ruanchen/workspace/learnML/Data/LogisticRegression/testSet.txt")
    for line in fr.readlines():
        lineArr = line.strip().split()
        train_x.append([1.0, float(lineArr[0]), float(lineArr[1])])
        train_y.append([float(lineArr[2])])
    return np.array(train_x), np.array(train_y)
    
if __name__ == "__main__":
    
    print "step 1: load data..."
    train_x, train_y = loadData()
    test_x = train_x
    test_y = train_y
    
    print "step 2: training..."
    # optimizeType
    # 1 : (batch) Gradient Descent
    # 2 : Stochastic Gradient Descent
    # 3: Modified Stochastic Gradient Descent
    opts = {'alpha': 0.01, 'maxIter': 1000, 'optimizeType': 2}
    optimalWeights = trainLogRegres(train_x, train_y, opts)
    
    print "step 3: testing..."
    accuracy = testLogRegres(optimalWeights, test_x, test_y)
    
    
    print "step 4: show result..."
    print "The classify accuracy is: %.3f%%" % (accuracy * 100)
    showLogRegres(optimalWeights, train_x, train_y)