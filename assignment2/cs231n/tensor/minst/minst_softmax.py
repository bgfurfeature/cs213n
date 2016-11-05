from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import numpy as np
import pandas as pd

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

print(len(mnist.test.images))


# Load the (preprocessed) data; csv file
def load_data(training_file, test_file):

    data_set = pd.read_csv(training_file)
    train_target = data_set[[0]].values.ravel()
    train = data_set.iloc[:, 1:].values
    test = pd.read_csv(test_file).values

    # convert to array, specify data type, and reshape
    X_train = np.array(train).reshape(42000, 28 * 28).astype(np.uint8)
    y_train = np.array(train_target).astype(np.uint8)
    # X_train = np.array(train).reshape((-1, 1, 28, 28)).astype(np.uint8)
    test = np.array(test).reshape(28000, 28 * 28).astype(np.uint8)

    return {
        'X_train': X_train, 'y_train': y_train,
        'X_test': test
    }


data_set_home='/mnt/hgfs/cs231n/cs231n/assignment2/cs231n/datasets/digtialRecognizer/'

data = load_data(data_set_home + "train.csv", data_set_home + "test.csv")

train_data = data['X_train']  # 42000
train_label = data['y_train']
test_data = data['X_test']  # 28000
print "train_data length: %d" % len(train_data)
print "test_data length: %d" % len(test_data)

sess = tf.InteractiveSession()

x = tf.placeholder(tf.float32, shape=[None, 784])  # None means whatever size you like
y_ = tf.placeholder(tf.float32, shape=[None, 10])

W = tf.Variable(tf.random_normal(shape=[784, 10]), name='W1')  #
b = tf.Variable(tf.zeros([10]), name='b1')


########################################################################################################################
# one layer fully connected network

# sess.run(tf.initialize_all_variables())
#
# y = tf.matmul(x,W) + b
#
# cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y, y_))
#
# train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
#
# for i in range(1000):
#   batch = mnist.train.next_batch(100)
#   train_step.run(feed_dict={x: batch[0], y_: batch[1]})
#
# correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
#
# accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
#
# print(accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels}))

#######################################################################################################################
#  convolution net work - same as LeNet
# LeNet try : two convlayer didn't work well, there are something wrong that i am not figured out                      #
# firstConv -> 28 x 28 x1 with filter 5 x 5 x 32, stride 1, pad 2 transformed to 28 x 28 x 32                          #
# maxpool ->  28 x 28 x 32 with pool size 2 x 2, stride 2 transformed to 14 x 14 x 32                                  #
# secondConv -> 14 x 14 x 32 with  filter 5 x 5 x 64, stride 1, pad 2 transformed to 14 x 14 x 64
# maxpool -> 14 x 14 x 64 with pool size 2 x 2, stride 2 transformed to 7 x 7 x 64
# fc_1 -> neuron number 1024
# fc_2 -> neuron number 84 # abandon
# dropout 0.5
# out -> classes 10
#
# init weight
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


# init bias
def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


# define convolution layer
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')  # Number -  Height - Weight - Channels


# define pooling layer
def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')


W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])

x_image = tf.reshape(x, [-1, 28, 28, 1])

h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])

h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

W_fc2 = weight_variable([1024, 84])
b_fc2 = bias_variable([84])

h_fc2 = tf.nn.relu(tf.matmul(h_fc1, W_fc2) + b_fc2)

keep_prob = tf.placeholder(tf.float32)
h_fc2_drop = tf.nn.dropout(h_fc2, keep_prob)

W_fc3 = weight_variable([84, 10])
b_fc3 = bias_variable([10])

y_conv = tf.matmul(h_fc2_drop, W_fc3) + b_fc3

########################################################################################################################
# define cal function for this model: the predict val is stored in y_conv,the correct label is stored in y_, all you need
# to do is to set the feed_dict
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y_conv, y_))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
predict_function = tf.argmax(y_conv, 1)
correct_prediction = tf.equal(predict_function, tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))  # Casts bool to a new type tf.float32
# define all is over!!
# start to run

sess.run(tf.initialize_all_variables())  # init all variables

for i in range(20001):
    num_train = train_data.shape[0]
    batch_mask = np.random.choice(num_train, 50)
    X_batch = train_data[batch_mask]
    X_batch_normal = np.multiply(X_batch, 1.0 / 255.0)
    y_batch = train_label[batch_mask]
    # batch = mnist.train.next_batch(50)
    if i % 100 == 0:
        train_accuracy = accuracy.eval(feed_dict={x:X_batch_normal, y_: y_batch, keep_prob: 1.0})
        print("step %d, training accuracy %g" % (i, train_accuracy))
    train_step.run(feed_dict={x:X_batch_normal, y_: y_batch, keep_prob: 0.5})

mnist_test = np.multiply(mnist.test.images, 1.0 / 255.0)

print("test accuracy %g" % accuracy.eval(feed_dict={x: mnist_test, y_: mnist.test.labels, keep_prob: 1.0}))

BATCH_SIZE = 50

# predict test set
# predicted_lables =  predict_function.eval(feed_dict={x: test_data, keep_prob: 1.0})

# using batches is more resource efficient
predicted_lables = np.zeros(test_data.shape[0])
for i in range(0,test_data.shape[0]//BATCH_SIZE)
    test_batch =  np.multiply(test_data[i * BATCH_SIZE : (i+1) * BATCH_SIZE], 1.0 / 255.0)
    predicted_lables[i * BATCH_SIZE : (i+1) * BATCH_SIZE] = predict_function.eval(feed_dict={x: test_batch, keep_prob: 1.0})
# save results
np.savetxt('LeNet_cnn_2.csv', np.c_[range(1, len(test_data) + 1), predicted_lables], delimiter=',',
           header='ImageId,Label', comments='', fmt='%s')
