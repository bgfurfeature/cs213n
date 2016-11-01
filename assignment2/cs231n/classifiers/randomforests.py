# coding=utf-8
"""
randon forest
"""
from __future__ import division
import numpy as np
import math


class node:
    """
     save dt structure for predict
    """
    def __init__(self, col=-1, value=None, results=None, trueBranch=None, falseBranch=None):
        self.col = col
        self.value = value
        self.results = results
        self.trueBranch = trueBranch
        self.falseBranch = falseBranch

    def getLabel(self):
        if self.results is None:
            return None
        else:
            max_counts = 0
            for key in self.results.keys():
                if self.results[key] > max_counts:
                    label = key
                    max_counts = self.results[key]
        return label


class RandomForestsClassifier:

    """
    classifier = randomforest.RandomForestsClassifier(n_bootstrapSamples=10)
    classifier.fit(training_data)
    """
    def __init__(self, n_bootstrapSamples=20):
        self.n_bootstrapSamples = n_bootstrapSamples
        self.list_tree = []

    def divideSet(self, samples, column, value):

        splitFunction = None
        if isinstance(value, int) or isinstance(value, float):
            splitFunction = lambda row: row[column] >= value
        else:
            splitFunction = lambda row: row[column] == value
        set1 = [row for row in samples if splitFunction(row)]  # true
        set2 = [row for row in samples if not splitFunction(row)]  # false
        return set1, set2

    # count number of each class: data is 2 dimension list
    def uniqueCounts(self, samples):
        results = {}
        for row in samples:
            r = row[len(row) - 1]  # the last col is the class
            if r not in results:
                results[r] = 0
            results[r] += 1
        return results

    # get gini(p)
    def giniEstimate(self, samples):
        if len(samples) == 0:
            return 0
        total = len(samples)
        counts = self.uniqueCounts(samples)
        gini = 0
        for target in counts:
            gini += pow(counts[target], 2)
        gini = 1 - gini / pow(total, 2)
        return gini

    # create tree
    def buildTree(self, samples):  # CART

        if len(samples) == 0:
            return node()
        print "sample:" + str(samples)
        currentGini = self.giniEstimate(samples)
        print "currentGini:" + str(currentGini)
        bestGain = 0  # leave gini is 0
        bestCriteria = None  # standard
        bestSets = None
        colCount = len(samples[0]) - 1  # feature numbers
        colRange = range(0, colCount)
        np.random.shuffle(colRange)  # get random range for sampling
        for col in colRange[0:int(math.ceil(math.sqrt(colCount)))]:  # get random features
            colValues = {}
            for row in samples:
                colValues[row[col]] = 1
            for value in colValues.keys():
                (set1, set2) = self.divideSet(samples, col, value)
                # according to feature divied
                dividGini = (len(set1) * self.giniEstimate(set1) + len(set2) * self.giniEstimate(set2)) / len(samples)
                gain = currentGini - dividGini
                print "col:" + str(col) + " , value:" + str(value) + " , dividedGini:" + str(dividGini) + " , gini lower result is:" + str(gain)
                if gain > bestGain and len(set1) > 0 and len(set2) > 0:
                    bestGain = gain
                    bestCriteria = (col, value)
                    print "bestCriteria:" + str(bestCriteria) + " , best gini is:" + str(bestGain)
                    bestSets = (set1, set2)
        if bestGain > 0:
            trueBranch = self.buildTree(bestSets[0])
            falseBranch = self.buildTree(bestSets[1])
            return node(col=bestCriteria[0], value=bestCriteria[1], trueBranch=trueBranch, falseBranch=falseBranch)
        else:
            return node(results=self.uniqueCounts(samples))  # get class

    # show tree
    def printTree(self, tree, indent='  '):  # show dt
        if tree.results is not None:
            print str(tree.results)
        else:
            print "col:" + str(tree.col) + ',' + "feature:" + str(tree.value) + ' ?'
            print indent + 'T->',
            self.printTree(tree.trueBranch, indent + '  ')
            print indent + 'F->',
            self.printTree(tree.falseBranch, indent + '  ')

    # predict tree
    def predict_tree(self, observation, tree):  # use dt to classify
        if tree.results is not None:
            return tree.getLabel()
        else:
            v = observation[tree.col]
            branch = None
            if isinstance(v, int) or isinstance(v, float):
                if v >= tree.value:
                    branch = tree.trueBranch
                else:
                    branch = tree.falseBranch
            else:
                if v == tree.value:
                    branch = tree.trueBranch
                else:
                    branch = tree.falseBranch
            return self.predict_tree(observation, branch)

    # get samples
    def generateBootstrapSamples(self, data):  # chose bootstraping
        samples = []
        for i in range(len(data)):
            samples.append(data[np.random.randint(len(data))])
        return samples

    # training
    def fit(self, data):  # create random forest
        for i in range(self.n_bootstrapSamples):
            samples = self.generateBootstrapSamples(data)
            currentTree = self.buildTree(samples)
            self.list_tree.append(currentTree)

    # random forest predict
    def predict_randomForests(self, observation):  # classify for data
        global finalResult
        results = {}
        for i in range(len(self.list_tree)):
            currentResult = self.predict_tree(observation, self.list_tree[i])
            if currentResult not in results:
                results[currentResult] = 0
            results[currentResult] += 1
        max_counts = 0
        for key in results.keys():
            if results[key] > max_counts:
                finalResult = key
                max_counts = results[key]
        return finalResult
