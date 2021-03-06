""" url - https://www.kaggle.com/c/digit-recognizer """

from time import time

import matplotlib.pyplot as plt

from cs231n.classifiers.CIFAR10NetWork import *
from cs231n.kaggle.CIFAR10.CIFAR10Solver import *
from cs231n.kaggle.CIFAR10.load_data import *

plt.rcParams['figure.figsize'] = (10.0, 8.0)  # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

# Load the (preprocessed) data.
data = get_CIFAR10_data()

train_data = data["X_train"]
print train_data.shape

# print train_data[0][0]  # 1 x 28 x 28

# draw_a_picture(train_data)

test_data = data["X_test"]
# print test_data[0][0]

t0 = time()

# model = conv_relu_max_pool_affine_relu_affineNet(weight_scale=0.001, hidden_dim=500, reg=0.001)
# use LeNet
model = CIFAR10NetWork(weight_scale=0.001, reg=0.001)

solver = Solver(model, data,
                num_epochs=10, batch_size=50,
                update_rule='adadelta',
                optim_config={
                    'learning_rate': 1e-3,
                },
                verbose=True, print_every=100)

solver.train()

t1 = time()

print 'train time: %fs' % (t1 - t0)

# get the best_params,can save this for test,otherwise train is too time-consumed. you can try!!
param = model.params

test_target = solver.predict(test_data)

N = test_target.size

class_list = []

class_dic = {0: "airplane", 1: "automobile", 2: "bird", 3: "cat", 4: "deer", 5: "dog", 6: "frog", 7: "horse", 8: "ship", 9: "truck"}

for index in xrange(N):
    pre_class = class_dic[test_target[index]]
    class_list.append(pre_class)

t2 = time()

print 'predict time: %fs' % (t2 - t1)

# save results
np.savetxt('submission_cnn.csv', np.c_[range(1, len(test_data) + 1), class_list], delimiter=',',
           header='id,label', comments='', fmt='%s')

plt.subplot(2, 1, 1)
plt.plot(solver.loss_history, 'o')
plt.xlabel('iteration')
plt.ylabel('loss')

plt.subplot(2, 1, 2)
plt.plot(solver.train_acc_history, '-o')
plt.plot(solver.val_acc_history, '-o')
plt.legend(['train', 'val'], loc='upper left')
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.show()
