import os
import numpy as np
from cs231n.kaggle.CIFAR10.load_data import *
# class_dic = {0: "airplane", 1: "automobile", 2: "bird", 3: "cat", 4: "deer", 5: "dog", 6: "frog", 7: "horse", 8: "ship",
#              9: "truck"}
#
# # save results
# scores = np.array([[0, 1, 2]
#                     ,[3, 4, 5],
#                       [6, 7, 8]])
#
# y_pred = np.argmax(scores, axis=1)
# print y_pred
# N = y_pred.size
# print N
#
# class_list = []
# for index in xrange(N):
#     pre_class = class_dic[y_pred[index]]
#     class_list.append(pre_class)
#
# np.savetxt('submission_cnn.csv', np.c_[range(1, len(scores) + 1), class_list], delimiter=',',
#            header='id,label', comments='', fmt='%s')
