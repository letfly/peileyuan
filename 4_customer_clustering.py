from numpy import *


def loadDataList(fileName):
    dataList = []
    fr = open(fileName)
    for line in fr.readlines():
        curLine = line.strip().split('\t')
        fltLine = map(float, curLine)
        dataList.append(fltLine)
    return dataList


def randCent(dataMat, k):
    n = shape(dataMat)[1]
    centroids = mat(zeros((k, n)))
    for j in range(n):
        minJ = min(dataMat[:, j])
        rangeJ = float(max(dataMat[:, j])-minJ)
        centroids[:, j] = minJ + rangeJ*random.rand(k, 1)
        print rangeJ, centroids
    return centroids

'''
if __name__ == '__main__':
    from numpy import *
    dataMat=mat(loadDataList('testSet.txt'))
    print min(dataMat[:,0])
    print min(dataMat[:,1])
    print max(dataMat[:,1])
    print max(dataMat[:,0])
    print randCent(dataMat, 2)
'''


def distEclud(vecA, vecB):
    return sqrt(sum(power(vecA - vecB, 2)))


def kMeans(dataMat, k, distMeas=distEclud, createCent=randCent):
    m = shape(dataMat)[0]
    clusterAssment = mat(zeros((m, 2)))
    centroids = createCent(dataMat, k)
    clusterChanged = True
    while clusterChanged:
        clusterChanged = False
        for i in range(m):
            minDist = inf
            minIndex = -1
            for j in range(k):
                distJI = distMeas(centroids[j, :], dataMat[i, :])
                if distJI < minDist:
                    minDist = distJI
                    minIndex = j
            if clusterAssment[i, 0] != minIndex:
                clusterChanged = True
            clusterAssment[i, :] = minIndex, minDist**2
        print centroids
        for cent in range(k):
            ptsInClust = dataMat[nonzero(clusterAssment[:, 0].A == cent)[0]]
            centroids[cent, :] = mean(ptsInClust, axis=0)
    return centroids, clusterAssment

if __name__ == '__main__':
    dataMat = mat(loadDataList('testSet.txt'))
    print kMeans(dataMat, 4)
