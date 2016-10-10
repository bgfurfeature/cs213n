# As usual, a bit of setup

import time
import numpy as np
import matplotlib.pyplot as plt
from cs231n.classifiers.fc_net import *
from cs231n.gradient_check import eval_numerical_gradient
from cs231n.solver import Solver
from cs231n.test.load_data import *

plt.rcParams['figure.figsize'] = (10.0, 8.0)  # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'


#############################################################################
# for auto-reloading external modules
# see http://stackoverflow.com/questions/1907993/autoreload-of-modules-in-ipython

def rel_error(x, y):
    """ returns relative error """
    return np.max(np.abs(x - y) / (np.maximum(1e-8, np.abs(x) + np.abs(y))))


#############################################################################
# LOAD DATA
# data = get_CIFAR10_data()  # test inside must have a py filename __init__.py otherwise it can not be referenced
# for k, v in data.iteritems():
#     print '%s: ' % k, v.shape
#
# X_test = data['X_test']
# y_test = data['y_test']
#
# X_val = data['X_val']
# y_val = data['y_val']
#############################################################################
# Batch normalization: Forward

# x = np.array([[0, 1, 0, 3],
#               [1, 0, 2, 0]])
# N,D = x.shape
#
# gamma = np.zeros(D,dtype=x.dtype)
# # print gamma
# beta = np.zeros(D,dtype=x.dtype)
# sample_mean = np.mean(x, axis=0)
# # print sample_mean
# sample_var = np.var(x, axis=0)
# print sample_var
# eps = 1e-5
# x_normalized = (x - sample_mean) / np.sqrt(sample_var + eps)
# # print x_normalized
# out = gamma * x_normalized + beta
# print out
#############################################################################
# Simulate the Batch normalization forward pass for a two-layer network

# N, D1, D2, D3 = 200, 50, 60, 3
# X = np.random.randn(N, D1)
# W1 = np.random.randn(D1, D2)
# W2 = np.random.randn(D2, D3)
# a = np.maximum(0, X.dot(W1)).dot(W2)
#
# print 'Before batch normalization:'
# print '  means: ', a.mean(axis=0)
# print '  stds: ', a.std(axis=0)
#
# # Means should be close to zero and stds close to one
# print 'After batch normalization (gamma=1, beta=0)'
# a_norm, _ = batchnorm_forward(a, np.ones(D3), np.zeros(D3), {'mode': 'train'})
# print '  mean: ', a_norm.mean(axis=0)
# print '  std: ', a_norm.std(axis=0)
#
# # Now means should be close to beta and stds close to gamma
# gamma = np.asarray([1.0, 2.0, 3.0])
# beta = np.asarray([11.0, 12.0, 13.0])
# a_norm, _ = batchnorm_forward(a, gamma, beta, {'mode': 'train'})
# print 'After batch normalization (nontrivial gamma, beta)'
# print '  means: ', a_norm.mean(axis=0)
# print '  stds: ', a_norm.std(axis=0)

#############################################################################
# Check the test-time forward pass by running the training-time
# forward pass many times to warm up the running averages, and then
# checking the means and variances of activations after a test-time
# forward pass.

# N, D1, D2, D3 = 200, 50, 60, 3
# W1 = np.random.randn(D1, D2)
# W2 = np.random.randn(D2, D3)
#
# bn_param = {'mode': 'train'}
# gamma = np.ones(D3)  # all element is 1
# beta = np.zeros(D3)  # D3 = 3 -> [0,0,0]
# for t in xrange(50):
#     X = np.random.randn(N, D1)
#     a = np.maximum(0, X.dot(W1)).dot(W2)
#     batchnorm_forward(a, gamma, beta, bn_param)
#
# bn_param['mode'] = 'test'
# X = np.random.randn(N, D1)
# a = np.maximum(0, X.dot(W1)).dot(W2)
# a_norm, _ = batchnorm_forward(a, gamma, beta, bn_param)
#
# # Means should be close to zero and stds close to one, but will be
# # noisier than training-time forward passes.
# print 'After batch normalization (test-time):'
# print '  means: ', a_norm.mean(axis=0)
# print '  stds: ', a_norm.std(axis=0)

#############################################################################
# Batch Normalization: backward
# Gradient check batchnorm backward pass

# N, D = 4, 5
# x = 5 * np.random.randn(N, D) + 12
# gamma = np.random.randn(D)
# beta = np.random.randn(D)
# dout = np.random.randn(N, D)
# print "dout"
# print dout
#
# bn_param = {'mode': 'train'}
# fx = lambda x: batchnorm_forward(x, gamma, beta, bn_param)[0]
# fg = lambda a: batchnorm_forward(x, gamma, beta, bn_param)[0]
# fb = lambda b: batchnorm_forward(x, gamma, beta, bn_param)[0]
#
# dx_num = eval_numerical_gradient_array(fx, x, dout)
# da_num = eval_numerical_gradient_array(fg, gamma, dout)
# db_num = eval_numerical_gradient_array(fb, beta, dout)
#
# _, cache = batchnorm_forward(x, gamma, beta, bn_param)
#
# dx, dgamma, dbeta = batchnorm_backward(dout, cache)
#
# # print "dx"
# # print dx
# # print "dgamma"
# # print dgamma
# # print "dbeta"
# # print dbeta
# print 'dx error: ', rel_error(dx_num, dx)
# print 'dgamma error: ', rel_error(da_num, dgamma)
# print 'dbeta error: ', rel_error(db_num, dbeta)

#############################################################################
# Batch Normalization: alternative backward

# N, D = 100, 500
# x = 5 * np.random.randn(N, D) + 12
# gamma = np.random.randn(D)
# beta = np.random.randn(D)
# dout = np.random.randn(N, D)
#
# bn_param = {'mode': 'train'}
# out, cache = batchnorm_forward(x, gamma, beta, bn_param)
#
# t1 = time.time()
# dx1, dgamma1, dbeta1 = batchnorm_backward(dout, cache)
# t2 = time.time()
# dx2, dgamma2, dbeta2 = batchnorm_backward_alt(dout, cache)
# t3 = time.time()
#
# print 'dx difference: ', rel_error(dx1, dx2)
# print 'dgamma difference: ', rel_error(dgamma1, dgamma2)
# print 'dbeta difference: ', rel_error(dbeta1, dbeta2)
# print 'speedup: %.2fx' % ((t2 - t1) / (t3 - t2))

#############################################################################
# Fully Connected Nets with Batch Normalization

# you should insert a batch normalization layer before each ReLU nonlinearity. The outputs from the last layer of the network should not be normalized.
# N, D, H1, H2, C = 2, 15, 20, 30, 10
# X = np.random.randn(N, D)
# y = np.random.randint(C, size=(N,))
#
# for reg in [0, 3.14]:
#     print 'Running check with reg = ', reg
#     model = FullyConnectedNet([H1, H2], input_dim=D, num_classes=C,
#                               reg=reg, weight_scale=5e-2, dtype=np.float64,
#                               use_batchnorm=True)
#
#     loss, grads = model.loss(X, y)
#     print 'Initial loss: ', loss
#
#     for name in sorted(grads):
#         f = lambda _: model.loss(X, y)[0]
#         grad_num = eval_numerical_gradient(f, model.params[name], verbose=False, h=1e-5)
#         print '%s relative error: %.2e' % (name, rel_error(grad_num, grads[name]))
#     if reg == 0:
#         print

#############################################################################
# Batchnorm for deep networks
# Try training a very deep net with batchnorm
hidden_dims = [100, 100, 100, 100, 100]

num_train = 1000
data = get_CIFAR10_data()
small_data = {
  'X_train': data['X_train'][:num_train],
  'y_train': data['y_train'][:num_train],
  'X_val': data['X_val'],
  'y_val': data['y_val'],
}

small_data = data

for k, v in data.iteritems():
     print '%s: ' % k, v.shape
weight_scale = 2e-2
for reg in [0.01, 0.1, 1.0]:

    bn_model = FullyConnectedNet(hidden_dims, reg=reg, weight_scale=weight_scale, use_batchnorm=True)
    model = FullyConnectedNet(hidden_dims, reg=reg, weight_scale=weight_scale, use_batchnorm=False)

    bn_solver = Solver(bn_model, small_data,
                    num_epochs=10, batch_size=50,
                    update_rule='adam',
                    optim_config={
                      'learning_rate': 1e-3,
                    },
                    verbose=True, print_every=200)
    bn_solver.train()

# solver = Solver(model, small_data,
#                 num_epochs=10, batch_size=50,
#                 update_rule='adam',
#                 optim_config={
#                   'learning_rate': 1e-3,
#                 },
#                 verbose=True, print_every=200)
# solver.train()
#############################################################################
# Batch normalization and initialization
# Try training a very deep net with batchnorm
# hidden_dims = [50, 50, 50, 50, 50, 50, 50]
#
# num_train = 1000
# small_data = {
#   'X_train': data['X_train'][:num_train],
#   'y_train': data['y_train'][:num_train],
#   'X_val': data['X_val'],
#   'y_val': data['y_val'],
# }
#
# bn_solvers = {}
# solvers = {}
# weight_scales = np.logspace(-4, 0, num=20)
# for i, weight_scale in enumerate(weight_scales):
#   print 'Running weight scale %d / %d' % (i + 1, len(weight_scales))
#   bn_model = FullyConnectedNet(hidden_dims, weight_scale=weight_scale, use_batchnorm=True)
#   model = FullyConnectedNet(hidden_dims, weight_scale=weight_scale, use_batchnorm=False)
#
#   bn_solver = Solver(bn_model, small_data,
#                   num_epochs=10, batch_size=50,
#                   update_rule='adam',
#                   optim_config={
#                     'learning_rate': 1e-3,
#                   },
#                   verbose=False, print_every=200)
#   bn_solver.train()
#   bn_solvers[weight_scale] = bn_solver
#
#   solver = Solver(model, small_data,
#                   num_epochs=10, batch_size=50,
#                   update_rule='adam',
#                   optim_config={
#                     'learning_rate': 1e-3,
#                   },
#                   verbose=False, print_every=200)
#   solver.train()
#   solvers[weight_scale] = solver
#############################################################################
#  get the best weight-scale initialization of plot the acc
# Plot results of weight scale experiment
# best_train_accs, bn_best_train_accs = [], []
# best_val_accs, bn_best_val_accs = [], []
# final_train_loss, bn_final_train_loss = [], []
#
# for ws in weight_scales:
#   best_train_accs.append(max(solvers[ws].train_acc_history))
#   bn_best_train_accs.append(max(bn_solvers[ws].train_acc_history))
#
#   best_val_accs.append(max(solvers[ws].val_acc_history))
#   bn_best_val_accs.append(max(bn_solvers[ws].val_acc_history))
#
#   final_train_loss.append(np.mean(solvers[ws].loss_history[-100:]))
#   bn_final_train_loss.append(np.mean(bn_solvers[ws].loss_history[-100:]))
#
# plt.subplot(3, 1, 1)
# plt.title('Best val accuracy vs weight initialization scale')
# plt.xlabel('Weight initialization scale')
# plt.ylabel('Best val accuracy')
# plt.semilogx(weight_scales, best_val_accs, '-o', label='baseline')
# plt.semilogx(weight_scales, bn_best_val_accs, '-o', label='batchnorm')
# plt.legend(ncol=2, loc='lower right')
#
# plt.subplot(3, 1, 2)
# plt.title('Best train accuracy vs weight initialization scale')
# plt.xlabel('Weight initialization scale')
# plt.ylabel('Best training accuracy')
# plt.semilogx(weight_scales, best_train_accs, '-o', label='baseline')
# plt.semilogx(weight_scales, bn_best_train_accs, '-o', label='batchnorm')
# plt.legend()
#
# plt.subplot(3, 1, 3)
# plt.title('Final training loss vs weight initialization scale')
# plt.xlabel('Weight initialization scale')
# plt.ylabel('Final training loss')
# plt.semilogx(weight_scales, final_train_loss, '-o', label='baseline')
# plt.semilogx(weight_scales, bn_final_train_loss, '-o', label='batchnorm')
# plt.legend()
#
# plt.gcf().set_size_inches(10, 15)
# plt.show()
#############################################################################
