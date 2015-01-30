import numpy as np
import matplotlib.pyplot as plt

def loadDataSet(fileName):
    dataMat = []
    fr = open(fileName)
    for line in fr.readlines():
        curLine = line.strip().split('\t')
        fltLine = map(float, curLine)
        dataMat.append(fltLine)
    return np.array(dataMat)

# calculate Euclidean distance
def euclDistance(vector1, vector2):
    return np.sqrt(np.sum(np.power(vector1 - vector2, 2)))

# initialize centroids by randomly sampling points within the range of 
# whole dataset                    
def initRandCentroids(dataSet, k):
    dim = np.shape(dataSet)[1]
    centroids = np.zeros((k, dim))
    for j in range(dim):
        minJ = min(dataSet[:, j])
        rangeJ = float(max(dataSet[:, j]) - minJ)
        centroids[:, j] = minJ + rangeJ * np.random.rand(k)
    return centroids

# initialize centroids by randomly sampling a point in the dataset
def initRandCentroids2(dataSet, k):
    numSamples, dim = dataSet.shape
    centroids = np.zeros((k, dim))
    for i in range(k):
        index = int(np.random.uniform(0, numSamples))
        centroids[i, :] = dataSet[index, :]
    return centroids

def initCentroidsKmeansPP(dataSet, k):
    numSamples = dataSet.shape[0]
    centroid0 = dataSet[int(np.random.uniform(0, numSamples)), :]
    centroids = [centroid0]
    while len(centroids) < k:
        dSquare = np.array([np.min([np.sum((sample - cent)**2) for cent in centroids]) for sample in dataSet])
        probs = dSquare / dSquare.sum()
        cumProbs = probs.cumsum()
        r = np.random.random()
        index = np.where(cumProbs >= r)[0][0]
        centroids.append(dataSet[index, :])
    return np.array(centroids)

def kMeans(dataSet, k, distMeas=euclDistance, createCent=initRandCentroids):
    numSamples = dataSet.shape[0]
    clusterAssment = np.zeros((numSamples,2))
    
    # step 1: initialize centroids
    centroids = createCent(dataSet, k)
    
    clusterChanged = True
    while clusterChanged:
        clusterChanged = False
        # for each sample
        for i in xrange(numSamples):
            minDist = np.inf
            minIndex = -1
            # for each centroid
            # step 2: find the cloest centroid
            for j in range(k):
                distance = distMeas(centroids[j,:], dataSet[i,:])
                if distance < minDist:
                    minDist = distance
                    minIndex = j
            # step 3: update its cluster
            if clusterAssment[i, 0] != minIndex:
                clusterChanged = True
            clusterAssment[i,:] = minIndex, minDist ** 2
        # step 4: update centroids
        for cent in range(k):
            ptsInCluster = dataSet[np.nonzero(clusterAssment[:, 0].T == cent)[0]]
            centroids[cent,:] = np.mean(ptsInCluster, axis = 0)
     
    return centroids, clusterAssment

def kMedoids(dataSet, k, distMeas=euclDistance, createCent=initRandCentroids):
    numSamples = dataSet.shape[0]
    clusterAssment = np.zeros((numSamples,2))
    
    # step 1: initialize centroids
    centroids = createCent(dataSet, k)
    
    clusterChanged = True
    while clusterChanged:
        clusterChanged = False
        # for each sample
        for i in xrange(numSamples):
            minDist = np.inf
            minIndex = -1
            # for each centroid
            # step 2: find the cloest centroid
            for j in range(k):
                distance = distMeas(centroids[j,:], dataSet[i,:])
                if distance < minDist:
                    minDist = distance
                    minIndex = j
            # step 3: update its cluster
            if clusterAssment[i, 0] != minIndex:
                clusterChanged = True
            clusterAssment[i,:] = minIndex, minDist ** 2
        # step 4: update centroids
        for cent in range(k):
            ptsInCluster = dataSet[np.nonzero(clusterAssment[:, 0].T == cent)[0]]
            # different from kMeans by getting median instead of mean
            numClusterSamples = ptsInCluster.shape[0]
            minSumDist = np.inf
            for i in xrange(numClusterSamples):
                sumDist = 0.
                for j in xrange(numClusterSamples):
                    if j != i:
                        sumDist += distMeas(ptsInCluster[i, :], ptsInCluster[j, :])
                if sumDist < minSumDist:
                    minSumDist = sumDist
                    minIndexInCluster = i        
            centroids[cent,:] = ptsInCluster[minIndexInCluster, :]
     
    return centroids, clusterAssment

def biKmeans(dataSet, k, distMeas=euclDistance, createCent=initRandCentroids):
    numSamples = dataSet.shape[0]
    clusterAssment = np.zeros((numSamples, 2))
    
    # step 1: the initial cluster is the whole dataset
    centroid0 = np.mean(dataSet, axis=0).tolist()[0]
    centList = [centroid0]
    for i in xrange(numSamples):
        clusterAssment[i, 1] = distMeas(np.array(centroid0), dataSet[i, :]) ** 2
        
    while len(centList) < k:
        minSSE = np.inf
        for i in range(len(centList)):
            # step 2: get samples in cluster i
            ptsInCluster = dataSet[np.nonzero(clusterAssment[:, 0].T == i)[0], :]
            
            # step 3: split it into 2 sub-clusters using kMeans
            centroidMat, splitClustAss = kMeans(ptsInCluster, 2, distMeas, createCent)
            
            # step 4: claculate  the sum of squre error after the split
            sseSplit = np.sum(splitClustAss[:, 1])
            sseNoSplit = np.sum(clusterAssment[np.nonzero(clusterAssment[:, 0].T != i)[0], 1])
            currSplitSSE = sseSplit + sseNoSplit
            
            print sseSplit, sseNoSplit
            
            # step 5: find the best split cluster which has the minimum SSE
            if currSplitSSE < minSSE:
                minSSE = currSplitSSE
                bestCentroidToSplit = i
                bestNewCentroid = centroidMat.copy()
                bestClusterAssment = splitClustAss.copy()
                
        # step 6: modify the cluster index for adding new cluster
        bestClusterAssment[np.nonzero(bestClusterAssment[:, 0].T == 1)[0], 0] = len(centList)
        bestClusterAssment[np.nonzero(bestClusterAssment[:, 0].T == 0)[0], 0] = bestCentroidToSplit
        
        print "the bestCentToSplit is: ", bestCentroidToSplit
        print "the len of bestClustAss is: ", len(bestClusterAssment)
        
        # step 7: update and append the centroids of the new 2 sub-clusters
        centList[bestCentroidToSplit] = bestNewCentroid[0, :]
        centList.append(bestNewCentroid[1, :])
        
        # step 8: update the index and error of the samples whose cluster have been changed
        clusterAssment[np.nonzero(clusterAssment[:, 0].T == bestCentroidToSplit)[0], :] = bestClusterAssment
        
    return np.array(centList), clusterAssment
            

def showCluster(dataSet, k, centroids, clusterAssment):
    numSamples, dim = dataSet.shape
    
    if dim != 2:
        print "Sorry! I can not draw because the dimension of your data is not 2!"
        return 1
    
    mark = ['or', 'ob', 'og', 'ok', '^r', '+r', 'sr', 'dr', '<r', 'pr']
    if k > len(mark):
        print "Sorry! Your k is too large!"
        return 1
    
    for i in xrange(numSamples):
        markIndex = int(clusterAssment[i, 0])
        plt.plot(dataSet[i, 0], dataSet[i, 1], mark[markIndex])
        
    mark = ['Dr', 'Db', 'Dg', 'Dk', '^b', '+b', 'sb', 'db', '<b', 'pb']
    
    for i in range(k):
        plt.plot(centroids[i, 0], centroids[i,1], mark[i], markersize = 12)
        
    plt.show()
        

if __name__ == "__main__":
    print "step 1: load data..."
    dataSet = loadDataSet("C:/Users/Ruanchen/workspace/learnML/Data/kMeans/testSet.txt")
    
    print "step 2: clustering..."
    k = 4
    #centroids, clusterAssment = kMeans(dataSet, k, createCent=initCentroidsKmeansPP)
    centroids, clusterAssment = biKmeans(dataSet, k, createCent=initCentroidsKmeansPP)
    #centroids, clusterAssment = kMedoids(dataSet, k, createCent=initRandCentroids2)
    
    print "step 3: show the result..."
    showCluster(dataSet, k, centroids, clusterAssment)
    
    

    