import numpy as np
import os


def createDataSet():
    group = np.array([[1.0, 0.9],
                      [1.0, 1.0],
                      [0.1, 0.2],
                      [0.0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels


def kNNClassify(newInput, dataSet, labels, k):
    
    numSamples = dataSet.shape[0]
    
    diff = np.tile(newInput, (numSamples, 1)) - dataSet
    squaredDiff = diff ** 2
    squaredDist = squaredDiff.sum(axis=1)
    distance = squaredDist ** 0.5
    
    sortedDistIndices = distance.argsort()
    
    classCount = {}
    for i in xrange(k):
        voteLabel = labels[sortedDistIndices[i]]
        classCount[voteLabel] = classCount.get(voteLabel, 0) + 1
        
    maxCount = 0
    for key, value in classCount.items():
        if value > maxCount:
            maxCount = value
            maxIndex = key
            
    return maxIndex

def img2vector(filename):
    rows = 32
    cols = 32
    imgVector = np.zeros((1, rows * cols))
    fileIn = open(filename)
    for row in xrange(rows):
        lineStr = fileIn.readline()
        for col in xrange(cols):
            imgVector[0, row * 32 + col] = int(lineStr[col])
    return imgVector

def loadDataSet():
    print '---Getting training set...'
    dataSetDir = 'C:/Users/Ruanchen/workspace/learnML/Data/digits/'
    traniningFileList = os.listdir(dataSetDir + 'trainingDigits')
    numSamples = len(traniningFileList)
    
    train_x = np.zeros((numSamples, 1024))
    train_y = []
    for i in xrange(numSamples):
        filename = traniningFileList[i]
        train_x[i, :] = img2vector(dataSetDir + 'trainingDigits/%s' % filename)
        label = int(filename.split('_')[0])
        train_y.append(label)
        
    print '---Getting testing set...'
    testingFileList = os.listdir(dataSetDir + 'testDigits')
    numSamples = len(testingFileList)
    test_x = np.zeros((numSamples, 1024))
    test_y = []
    for i in xrange(numSamples):
        filename = testingFileList[i]
        test_x[i, :] = img2vector(dataSetDir + 'testDigits/%s' % filename)
        label = int(filename.split('_')[0])
        test_y.append(label)
        
    return train_x, train_y, test_x, test_y

def testHandWriting(k):
    print 'step 1: load data...'
    train_x, train_y, test_x, test_y = loadDataSet()
    
    print 'step 2: training...'
    pass

    print 'step 3: testing...'
    numTestSamples = test_x.shape[0]
    matchCount = 0
    for i in xrange(numTestSamples):
        predict = kNNClassify(test_x[i], train_x, train_y, k)
        if predict == test_y[i]:
            matchCount += 1
    accuracy = float(matchCount) / numTestSamples
    
    print 'step 4: show the result...'
    print 'The classify accuracy is : %.2f%%' % (accuracy * 100)    
    
if __name__ == '__main__':
    
    #dataSet, labels = createDataSet()
    #testX = np.array([1.2, 1.0])
    #k = 3
    #outputLabel = kNNClassify(testX, dataSet, labels, 3)
    #print "Your input is: ", testX, " and classified to class: ", outputLabel
    
    #testX = np.array([0.1, 0.3])
    #outputLabel = kNNClassify(testX, dataSet, labels, 3)
    #print "Your input is: ", testX, " and classified to class: ", outputLabel
    k = 3
    testHandWriting(k)
    
    