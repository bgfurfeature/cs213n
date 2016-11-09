__author__ = 'CHAOJIANG'
import numpy as np
from cs231n.classifiers import *
from cs231n import *

import numpy as np
import cPickle as pickle
import os

# def load_CIFAR_batch(filename):
#   """ load single batch of cifar """
#   with open(filename, 'rb') as f:
#     datadict = pickle.load(f)
#     X = datadict['data']
#     Y = datadict['labels']
#     X = X.reshape(10000, 3, 32, 32).transpose(0,2,3,1).astype("float")
#     Y = np.array(Y)
#     return X, Y
#
# def load_CIFAR10(ROOT):
#   """ load all of cifar """
#   xs = []
#   ys = []
#   for b in range(1,6):
#     f = os.path.join(ROOT, 'data_batch_%d' % (b, ))
#     X, Y = load_CIFAR_batch(f)
#     xs.append(X)
#     ys.append(Y)
#   Xtr = np.concatenate(xs)
#   Ytr = np.concatenate(ys)
#   del X, Y
#   Xte, Yte = load_CIFAR_batch(os.path.join(ROOT, 'test_batch'))
#   return Xtr, Ytr, Xte, Yte
#
# Xtr, Ytr, Xte, Yte = load_CIFAR10("J:\github\dataSet\cifar-10-python\cifar-10-batches-py")
#
# print  Xte

W = np.random.randn(5, 3) * 0.1
print "W"
print W
X_dev = np.array([[1, 0, 1, 0, 1], [0, 1, 0, 1, 0], [0, 1, 1, 0, 1], [1, 0, 0, 1, 0]])
print "x_dev"
print X_dev
y_dev = np.array([0, 2, 0, 1])
loss, grad = svm_loss_naive(W, X_dev, y_dev, 1)
print 'loss: %f' % (loss)
