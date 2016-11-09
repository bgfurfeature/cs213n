__author__ = 'CHAOJIANG'
import numpy as np
x = np.array([[2, 0, 1, 0, 1], [0, 1, 0, 1, 0], [0, 1, 1, 0, 1], [1, 0, 0, 1, 0]])
print  "x"
print  x

# print np.max(x, axis = 1)
# shift_scores = x - np.max(x, axis = 1).reshape(-1, 1)
# print "shift_scores"
# print shift_scores
#
# down = np.sum(np.exp(shift_scores), axis=1).reshape(-1, 1)
# up = np.exp(shift_scores)
# softmax_cout = up/down
# print "softmax_cout"
# print softmax_cout
# N,D = x.shape
# y=[0,1,2,3]
# print softmax_cout[range(N), list(y)]
# temp = -np.sum(np.log(softmax_cout[range(N), list(y)]))
# print temp

# x_x = x*x
# print "x*x"
# print  x_x
#
# x_x = np.sum(x*x, axis = 1)
# print "x_x"
# print  x_x
# x_shape = np.reshape(np.sum(x*x, axis = 1),(-1,1))
# print "x_shape"
# print x_shape
#
# x_tr = np.reshape(np.sum(x*x, axis = 1),(1,-1))
# print "x_tr"
# print  x_tr
#
# print "+"
# print  x_shape + x_tr
#
# trw = 2*np.dot(x,np.transpose(x))
# print "trw"
# print x_shape + x_tr - trw




