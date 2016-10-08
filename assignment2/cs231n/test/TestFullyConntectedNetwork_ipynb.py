# As usual, a bit of setup

import time
import numpy as np
import matplotlib.pyplot as plt
from cs231n.classifiers.fc_net import *
from cs231n.gradient_check import eval_numerical_gradient, eval_numerical_gradient_array
from cs231n.solver import Solver

plt.rcParams['figure.figsize'] = (10.0, 8.0)  # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'


# for auto-reloading external modules
# see http://stackoverflow.com/questions/1907993/autoreload-of-modules-in-ipython

def rel_error(x, y):
    """ returns relative error """
    return np.max(np.abs(x - y) / (np.maximum(1e-8, np.abs(x) + np.abs(y))))


##  Load the (preprocessed) CIFAR10 data.

# data = get_CIFAR10_data()
# for k, v in data.iteritems():
#     print '%s: ' % k, v.shape

# Test the affine_forward function

# num_inputs = 2
# input_shape = (4, 5, 6)
# output_dim = 3
#
# input_size = num_inputs * np.prod(input_shape)
# print "input_sizeL:%d" % input_size
#
# weight_size = output_dim * np.prod(input_shape)
# print "weight_size:%d" % weight_size
#
# x = np.linspace(-0.1, 0.5, num=input_size).reshape(num_inputs, *input_shape)  # data range(-0.1,0.5)
# # print x  # [2 x [4 x [5 x 6]]]
# w = np.linspace(-0.2, 0.3, num=weight_size).reshape(np.prod(input_shape), output_dim)   # data range(-0.2,0.3)
# # print w  # [120 x 3]
# b = np.linspace(-0.3, 0.1, num=output_dim)
# # print b  # [-0.3 -0.1  0.1]
#
# out, _ = affine_forward(x, w, b)
# correct_out = np.array([[1.49834967, 1.70660132, 1.91485297],
#                         [3.25553199, 3.5141327, 3.77273342]])

# Compare your output with ours. The error should be around 1e-9.
# print 'Testing affine_forward function:'
# print 'difference: ', rel_error(out, correct_out)

# Test the affine_backward function

x = np.random.randn(10, 2, 3)  # [10 x 6]
# print x  # [10 x [2 x 3]]
w = np.random.randn(6, 5)  # [6 x 5]
b = np.random.randn(5)  # [1 x 5]
# print b
dout = np.random.randn(10, 5)   # [10 x 5]

# print  x
dx_num = eval_numerical_gradient_array(lambda x: affine_forward(x, w, b)[0], x, dout)

# dw_num = eval_numerical_gradient_array(lambda w: affine_forward(x, w, b)[0], w, dout)
# db_num = eval_numerical_gradient_array(lambda b: affine_forward(x, w, b)[0], b, dout)
#
# _, cache = affine_forward(x, w, b)
# dx, dw, db = affine_backward(dout, cache)
#
# # The error should be around 1e-10
# print 'Testing affine_backward function:'
# print 'dx error: ', rel_error(dx_num, dx)
# print 'dw error: ', rel_error(dw_num, dw)
# print 'db error: ', rel_error(db_num, db)
