# coding=utf-8
"""
randon forest

data format: [[0,0,0,0,class1],[0,0,0,1,class2]....]
the last col is class
"""
from __future__ import division
import numpy as np
import math


# coding: utf-8
def data_correct(data, results='results'):
    final_dic = {}
    dict_ = data_format(data)
    results_dic = {}
    if dict_[results] != 'None':
        tmp_results_dic = data_format(dict_[results], regx1=',', regx2=':')
        for k, v in tmp_results_dic.iteritems():
            results_dic[float(k)] = int(v)
        dict_[results] = results_dic
    else:
        dict_[results] = None
    for k, v in dict_.iteritems():
        if k == 'col':
            final_dic[k] = int(v)
        elif k == 'value':
            if v != 'None':
                final_dic[k] = float(v)
            else:
                final_dic[k] = (v)
        else:
            final_dic[k] = v
    return final_dic


def data_format(data, regx1='|', regx2='='):
    dic = {}
    data_list = data.replace("{", "").replace("}", "").split(regx1)

    for d in data_list:
        k, v = d.split(regx2)
        dic[k] = v
    return dic


class Tree:
    def __init__(self, vertex, left, right):
        self.root = vertex
        self.root.trueBranch = left
        self.root.falseBranch = right


class node:
    """
     save dt structure for predict
    """

    def __init__(self, col=-1, results=None, value=None, trueBranch=None, falseBranch=None):
        self.col = col
        self.value = value
        self.results = results
        self.trueBranch = trueBranch
        self.falseBranch = falseBranch

    def getLabel(self):
        global label
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
        self.data_format = []
        self.var_pre_order = {}
        self.var_order = {}

    def construct_tree(self, pre_order, mid_order):
        # 忽略参数合法性判断
        if len(pre_order) == 0:
            return None
        # 前序遍历的第一个结点一定是根结点
        root_data = pre_order[0]
        root_node = node(col=(root_data['col']), value=(root_data['value']), results=root_data['results'], trueBranch=root_data['trueBranch'], falseBranch=root_data['falseBranch'])

        for i in range(0, len(mid_order)):
            vertex = mid_order[i]
            # min_node = node(col=vertex['col'], value=vertex['value'], results=vertex['results'], trueBranch=vertex['trueBranch'], falseBranch=vertex['falseBranch'])
            if vertex == root_data:
                break
        # 递归构造左子树和右子树
        left = self.construct_tree(pre_order[1: 1 + i], mid_order[:i])
        right = self.construct_tree(pre_order[1 + i:], mid_order[i + 1:])
        root_node.trueBranch = left
        root_node.falseBranch = right
        return root_node

    def load_model(self, n_bootstrapSamples=20):
        tree_list = {}
        for i in range(n_bootstrapSamples):
            f = open(str(i), 'r+')
            lines = f.readlines()
            pre_order_data = lines[0].split(',')
            order_data = lines[1].split(',')
            pre_order = [data_correct(data) for data in pre_order_data if len(data) > 5]
            order = [data_correct(data) for data in order_data if len(data) > 5]
            tree_list[i] = self.construct_tree(pre_order, order)
        return tree_list

    def save_model(self, list_tree={}):
        for i in range(len(self.list_tree)):
            self.data_format = []
            self.pre_order(self.list_tree[i])
            self.var_pre_order[i] = self.data_format

            self.data_format = []
            self.order(self.list_tree[i])
            self.var_order[i] = self.data_format

        for k, v in self.var_pre_order.iteritems():
            f = file(str(k), 'a+', 1024)
            for value in v:
                f.write(value + ",")
            f.write("\n")
            f.close()
        for k, v in self.var_order.iteritems():
            f = file(str(k), 'a+', 1024)
            for value in v:
              f.write(value + ",")
            f.write("\n")
            f.close()

    def order(self, tree):
        if tree.results is not None:
            data = "col=" + str(tree.col) + '|' + "value=" + str(tree.value) + '|results=' + str(
                tree.results) + '|trueBranch=' + str(tree.trueBranch) + '|falseBranch=' + str(tree.falseBranch)
            self.data_format.append(data)
            return
        self.order(tree.trueBranch)
        if tree.results is not None:
            data = "col=" + str(tree.col) + '|' + "value=" + str(tree.value) + '|results=' + str(
                tree.results) + '|trueBranch=' + str(tree.trueBranch) + '|falseBranch=' + str(tree.falseBranch)
            self.data_format.append(data)

        else:
            data = "col=" + str(tree.col) + '|' + "value=" + str(tree.value) + '|results=' + str(
                tree.results) + '|trueBranch=' + str(None) + '|falseBranch=' + str(None)
            self.data_format.append(data)

        self.order(tree.falseBranch)

    def pre_order(self, tree):
        if tree.results is not None:
            data = "col=" + str(tree.col) + '|' + "value=" + str(tree.value) + '|results=' + str(
                tree.results) + '|trueBranch=' + str(tree.trueBranch) + '|falseBranch=' + str(tree.falseBranch)
            self.data_format.append(data)

            return
        else:
            data = "col=" + str(tree.col) + '|' + "value=" + str(tree.value) + '|results=' + str(
                tree.results) + '|trueBranch=' + str(None) + '|falseBranch=' + str(None)
            self.data_format.append(data)

        self.pre_order(tree.trueBranch)
        self.pre_order(tree.falseBranch)

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
                print "col:" + str(col) + " , value:" + str(value) + " , dividedGini:" + str(
                    dividGini) + " , gini lower result is:" + str(gain)
                if gain > bestGain and len(set1) > 0 and len(set2) > 0:
                    bestGain = gain
                    bestCriteria = (col, value)
                    print "bestCriteria:" + str(bestCriteria) + " , best gini is:" + str(bestGain)
                    bestSets = (set1, set2)
        if bestGain > 0:
            print "create trueBranch child tree!!"
            trueBranch = self.buildTree(bestSets[0])
            print "create falseBranch child tree!!"
            falseBranch = self.buildTree(bestSets[1])
            print "return node!!"
            return node(col=bestCriteria[0], value=bestCriteria[1], trueBranch=trueBranch, falseBranch=falseBranch)
        else:
            print "saving result for node!!"
            return node(results=self.uniqueCounts(samples))  # get class

    # show tree
    def printTree(self, tree, indent='  '):  # show dt
        if tree.results is not None:
            print str(tree.results)
        else:
            print "col:" + str(tree.col) + '，' + "feature:" + str(tree.value) + ' ?'
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
