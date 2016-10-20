""" url - https://www.kaggle.com/c/digit-recognizer """

from time import time
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from cs231n.classifiers.DigitRecognizerConvNet import *
from cs231n.kaggle.DigitRecognizerConvNetSolver import Solver
import matplotlib.cm as cm

plt.rcParams['figure.figsize'] = (10.0, 8.0)  # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'


# draw the picture from a pixel matrix
def draw_a_picture(data_matrix):
    plt.imshow(data_matrix[0][0], cmap=cm.binary)
    plt.show()


# Load the (preprocessed) data.
def load_data(filename):
    data_set = pd.read_csv(filename + "\\train.csv")
    train_target = data_set[[0]].values.ravel()
    train = data_set.iloc[:, 1:].values
    test = pd.read_csv(filename + "\\test.csv").values

    # convert to array, specify data type, and reshape
    X_train = np.array(train).reshape(42000, 1, 28, 28).astype(np.uint8)
    y_train = np.array(train_target).astype(np.uint8)
    # X_train = np.array(train).reshape((-1, 1, 28, 28)).astype(np.uint8)
    test = np.array(test).reshape(28000, 1, 28, 28).astype(np.uint8)

    num_training = 41000
    num_validation = 1000

    mask = range(num_training, num_training + num_validation)
    X_val = X_train[mask]
    y_val = y_train[mask]
    mask = range(num_training)
    X_train = X_train[mask]
    y_train = y_train[mask]

    return {
        'X_train': X_train, 'y_train': y_train,
        'X_val': X_val, 'y_val': y_val,
        'X_test': test
    }


data = load_data("F:\SmartData-X\DataSet\KaggleCompetition\Digit Recongizer")

train_data = data["X_train"]
print train_data.shape

# print train_data[0][0]  # 1 x 28 x 28

# draw_a_picture(train_data)

test_data = data["X_test"]
# print test_data[0][0]

t0 = time()

# model = conv_relu_max_pool_affine_relu_affineNet(weight_scale=0.001, hidden_dim=500, reg=0.001)
# use LeNet
model = LeNet(weight_scale=0.001, reg=0.001)

solver = Solver(model, data,
                num_epochs=5, batch_size=50,
                update_rule='adam',
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

t2 = time()

print 'predict time: %fs' % (t2 - t1)

# save results
np.savetxt('submission_cnn.csv', np.c_[range(1, len(test_data) + 1), test_target], delimiter=',',
           header='ImageId,Label', comments='', fmt='%d')

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
