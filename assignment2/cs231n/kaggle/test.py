import os


class test(object):

    def __init__(self):

        self.params = {}

    # params = {'w1': 0.9, 'w2': 0.87}
    # f = open('model.txt', 'a')
    # for k, v in params.iteritems():
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

test.load_model("model.txt")