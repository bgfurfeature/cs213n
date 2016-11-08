from numpy import *


# load data
def load_data(fileName, delim=','):
    f = open(fileName)
    stringArr = [line.strip().split(delim) for line in f.readlines()]
    datArr = [map(float, item) for item in stringArr]
    return mat(datArr)


class pca(object):
    """
        PCA implementing
    """
    def pca(self, dataMatrix, top_n_feature=9999999):
        meanVals = mean(dataMatrix, axis=0)
        meanRemoved = dataMatrix - meanVals
        variences = var(meanRemoved, axis=0)
        pre_data = meanRemoved / variences
        covMat = cov(pre_data, rowvar=0)
        eigvals, eigVects = linalg.eig(mat(covMat))
        eigValInd = argsort(eigvals)  # sort goes smallest to largest
        eigValInd = eigValInd[:-(top_n_feature + 1): -1]  # set k = top_n_feature dimension
        selectVects = eigVects[:, eigValInd]  #
        lowDimDataMat = pre_data * selectVects
        return lowDimDataMat


if __name__ == '__main__':
    mata = load_data('data')
    a = pca().pca(mata, 1)
    print a
