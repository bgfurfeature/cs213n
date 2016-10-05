__author__ = 'CHAOJIANG'
import numpy as np
x = np.array([[2, 0, 1, 0, 1], [0, 1, 0, 1, 0], [0, 1, 1, 0, 1], [1, 0, 0, 1, 0]])
print  "x"
print  x
x_x = x*x
print "x*x"
print  x_x

x_x = np.sum(x*x, axis = 1)
print "x_x"
print  x_x
x_shape = np.reshape(np.sum(x*x, axis = 1),(-1,1))
print "x_shape"
print x_shape

x_tr = np.reshape(np.sum(x*x, axis = 1),(1,-1))
print "x_tr"
print  x_tr

print "+"
print  x_shape + x_tr

trw = 2*np.dot(x,np.transpose(x))
print "trw"
print x_shape + x_tr - trw




