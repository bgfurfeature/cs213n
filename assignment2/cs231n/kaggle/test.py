import os
import numpy as np


class test(object):

    def __init__(self):

        self.params = {}

        # w1 = np.array([[0, 1, 2], [3, 4, 5]])
        #
        # w2 = np.random.normal(0, 1e-2, (7, 1, 2, 2))
        #
        # # params = {'w1': 0.9, 'w2': 0.87}
        # self.params["w1"] = w1.copy()
        # self.params['w2'] = w2
        # f = open('model.txt', 'a')
        # for k, v in self.params.iteritems():
        #     f.write(k + "=" + str(v) + "\n")
        # f.close()

    def load_model(self, model_file):
        f = open(model_file, "r")
        lines = f.readlines()
        for item in lines:
            k, v = item.split("=")
            self.params[k] = float(v)
        f.close()
        print self.params


test = test()

# test.load_model("model.txt")
