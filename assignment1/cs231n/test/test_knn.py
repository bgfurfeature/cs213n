__author__ = 'CHAOJIANG'
import numpy as np
from cs231n.classifiers.k_nearest_neighbor import *

x_train = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
y_train = np.array([[1],[0],[1]])

knn = KNearestNeighbor()
knn.train(x_train, y_train)

x = np.array([[1,1,0]])
#dist_one_loop = knn.compute_distances_one_loop(x)
#print dist_one_loop

dist_two_loop = knn.compute_distances_two_loops(x)
print dist_two_loop

#res = knn.predict(x,k=1,num_loops= 1)
#print  res

label = knn.predict_labels(dist_two_loop,k=3)
print  label