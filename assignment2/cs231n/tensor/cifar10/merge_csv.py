import numpy as np
import pandas as pd


def load_data(file_num):
    result = []
    for i in range(1, file_num + 1):
        data_set = pd.read_csv("cifar10_CNN_%d.csv" % i )
        label = data_set.iloc[:10000, 1].values
        for item in label:
            result.append(item)

        # print result
        # print len(result)
    return  result

lable = load_data(30)
N = len(lable)
np.savetxt('cifar10_CNN.csv', np.c_[range(1, N + 1), lable], delimiter=',', header='id,label',
                       comments='', fmt='%s')
