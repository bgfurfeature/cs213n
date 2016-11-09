# pylint: disable=missing-docstring
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import gzip
import os
import re
import sys
import tarfile
from cs231n.load_data import *
import numpy as np
import tensorflow as tf


#
import time


class cifarTrain(object):

    def __init__(self):
        self.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 10000
        self.desc = "This is a train for specific traning data CIFAR-10!!"
        # Constants describing the training process.
        self.MOVING_AVERAGE_DECAY = 0.9999     # The decay to use for the moving average.
        self.NUM_EPOCHS_PER_DECAY = 350.0      # Epochs after which learning rate decays.
        self.LEARNING_RATE_DECAY_FACTOR = 0.1  # Learning rate decay factor.
        self.INITIAL_LEARNING_RATE = 0.1       # Initial learning rate.
        self.batch_size = 128

    def train(self, total_loss, global_step=0.0):

        """Train CIFAR-10 model.
        :param total_loss:
        :param global_step:
        """
        # Variables that affect learning rate.
        num_batches_per_epoch = self.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN / self.batch_size
        decay_steps = int(num_batches_per_epoch * self.NUM_EPOCHS_PER_DECAY)

        # Decay the learning rate exponentially based on the number of steps.
        lr = tf.train.exponential_decay(self.INITIAL_LEARNING_RATE,
                                          global_step,
                                          decay_steps,
                                          self.LEARNING_RATE_DECAY_FACTOR,
                                          staircase=True)
        tf.scalar_summary('learning_rate', lr)

        # Generate moving averages of all losses and associated summaries.
        # Compute the moving average of all individual losses and the total loss.
        loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
        losses = tf.get_collection('losses')
        loss_averages_op = loss_averages.apply(losses + [total_loss])

        # Attach a scalar summary to all individual losses and the total loss; do the
        # same for the averaged version of the losses.
        for l in losses + [total_loss]:
            # Name each loss as '(raw)' and name the moving average version of the loss
            # as the original loss name.
            tf.scalar_summary(l.op.name +' (raw)', l)
            tf.scalar_summary(l.op.name, loss_averages.average(l))

        # Compute gradients.
        with tf.control_dependencies([loss_averages_op]):
            opt = tf.train.GradientDescentOptimizer(lr)
            grads = opt.compute_gradients(total_loss)

        # Apply gradients.
        apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

        # Add histograms for trainable variables.
        for var in tf.trainable_variables():
            tf.histogram_summary(var.op.name, var)

        # Add histograms for gradients.
        for grad, var in grads:
            if grad is not None:
                tf.histogram_summary(var.op.name + '/gradients', grad)

        # Track the moving averages of all trainable variables.
        variable_averages = tf.train.ExponentialMovingAverage(self.MOVING_AVERAGE_DECAY, global_step)
        variables_averages_op = variable_averages.apply(tf.trainable_variables())

        with tf.control_dependencies([apply_gradient_op, variables_averages_op]):
            train_op = tf.no_op(name='train')

        return train_op


#
class cifarEvaluation(object):

    def __init__(self):
        self.desc = "This is a Evaluation for specific traning data CIFAR-10!!"


#
class cifarModel(object):

    def __init__(self, flag=None, batch_size=50, NUM_CLASSES=10):
        self.desc = "This is a cnn model for specific traning data CIFAR-10!!"
        self.flag = flag
        self.batch_size = batch_size
        self.NUM_CLASSES = NUM_CLASSES

    def dtype(self):
        if self.flag is not None:
            return tf.float32
        else:
            return tf.float16

    def bias(self, name, shape, value):
        initial = tf.constant_initializer(value)
        return tf.get_variable(initializer=initial, name=name, shape=shape, dtype=self.dtype())

    def _variable_with_weight_decay(self, name, shape, stddev, wd):
        """
        Args:
        name: name of the variable
        shape: list of ints
        stddev: standard deviation of a truncated Gaussian
        wd: add L2Loss weight decay multiplied by this float. If None, weight
            decay is not added for this Variable.
        Returns:
            Variable Tensor
        """
        dtype = self.dtype()

        initialer = tf.truncated_normal_initializer(stddev=stddev, dtype=dtype)
        var = tf.get_variable(name=name, shape=shape, dtype=dtype, initialer=initialer)

        if wd is not None:
            # weight l2 loss
            weight_deacy = tf.mul(tf.nn.l2_loss(var), wd, name='weight_losse')
            tf.add_to_collection('losses', weight_deacy)

        return var

    # BUILD network
    def process(self, image):

        """Build the CIFAR-10 model.
        Args:
            images: Images returned from distorted_inputs() or inputs().
        Returns:
            Logits.
        :param image:
        :return:
        """
        # conv1
        with tf.variable_scope('conv1') as scope:
            kernel = self._variable_with_weight_decay('weights', [5, 5, 3, 64], stddev=5e-2, wd=0.0)
            conv = tf.nn.conv2d(image, kernel, strides=[1, 1, 1, 1], padding='SAME')
            biases = self.bias(name='biases', shape=[64], value=0.0)
            bias = tf.nn.bias_add(conv, biases)
            conv1 = tf.nn.relu(bias, name=scope.name)

        # pool1
        pool1 = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')

        # norm1
        norm1 = tf.nn.lrn(pool1, depth_radius=4, bias=1.0, alpha=1e-2 / 9.0, beta=0.75, name='norm1')

        # conv2
        with tf.variable_scope('conv2') as scope:
            kernel = self._variable_with_weight_decay('weights', shape=[5, 5, 64, 64], stddev=5e-2, wd=0.0)
            conv = tf.nn.conv2d(norm1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = self.bias('biases', [64], value=0.1)
            bias = tf.nn.bias_add(conv, biases)
            conv2 = tf.nn.relu(bias, name=scope.name)

        # norm2
        norm2 = tf.nn.lrn(conv2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm2')

        # pool2
        pool2 = tf.nn.max_pool(norm2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool2')

        # local3
        with tf.variable_scope('local3') as scope:
            # Move everything into depth so we can perform a single matrix multiply.
            reshape = tf.reshape(pool2, [self.batch_size, -1])
            dim = reshape.get_shape()[1].value
            weights = self._variable_with_weight_decay('weights', shape=[dim, 384], stddev=0.04, wd=0.004)
            biases = self.bias('biases', [384], value=0.1)
            local3 = tf.nn.relu(tf.matmul(reshape, weights) + biases, name=scope.name)

        # local4
        with tf.variable_scope('local4') as scope:
            weights = self._variable_with_weight_decay('weights', shape=[384, 192], stddev=0.04, wd=0.004)
            biases = self.bias('biases', [192], value=0.1)
            local4 = tf.nn.relu(tf.matmul(local3, weights) + biases, name=scope.name)

        # softmax,softmax(WX + b)
        with tf.variable_scope('softmax_linear') as scope:
            weights = self._variable_with_weight_decay('weights', [192, self.NUM_CLASSES], stddev=1 / 192.0, wd=0.0)
            biases = self.bias('biases', [ self.NUM_CLASSES], value=0.0)
            softmax_linear = tf.add(tf.matmul(local4, weights), biases, name=scope.name)

        return softmax_linear

    def loss(self, logits, labels):
        """Add L2Loss to all the trainable variables.
          Add summary for "Loss" and "Loss/avg".
          Args:
            logits: Logits from inference().
            labels: Labels from distorted_inputs or inputs(). 1-D tensor
                    of shape [batch_size]
          Returns:
            Loss tensor of type float.
        :param logit:
        :param labels:
        :return:
        """
        labels = tf.cast(labels, tf.int64)
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits, labels, name='cross_entropy_per_example')
        cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
        tf.add_to_collection('losses', cross_entropy_mean)  # data loss

        # The total loss is defined as the cross entropy loss plus all of the weight
        # decay terms (L2 loss).
        # data loss + weight loss
        return tf.add_n(tf.get_collection('losses'), name='total_loss')


# 0 => [1 0 0 0 0 0 0 0 0 0]
# 1 => [0 1 0 0 0 0 0 0 0 0]
# ...
# 9 => [0 0 0 0 0 0 0 0 0 1]
def label_to_one_hot(labels, num_classes):
    num_batch = labels.shape[0]
    index_offset = np.arange(
        num_batch) * num_classes  # each row has num_classes item，so for each row the offset is row_number * num_classes
    label_flat_hot = np.zeros(num_batch, num_classes)
    label_flat_hot.flat[index_offset + labels.ravel()] = 1
    return label_flat_hot

if __name__ == '__main__':

    cifarModel = cifarModel()
    global_step = tf.Variable(0, trainable=False)
    # load_data
    data = get_CIFAR10_data('/mnt/hgfs/cs231n/cs231n/assignment2/cs231n/datasets/cifar-10-batches-py')
    train_data = data['X_train']
    train_labels = data['y_train']
    test_data = data['X_test']
    test_label = data['y_test']
    label_count = np.unique(train_labels).shape[0]

    images = tf.placeholder(tf.float32, shape=[None, 32, 32, 3])  # None means whatever size you like
    labels = tf.placeholder(tf.float32, shape=[None, cifarModel.NUM_CLASSES])

    # graph
    session = tf.InteractiveSession
    init = tf.initialize_all_variables()
    # x_image = tf.reshape(images, [-1, 32, 32, 3])
    logits = cifarModel.process(images)

    # Calculate loss.
    loss = cifarModel.loss(logits, labels)
    train_step = tf.train.AdadeltaOptimizer(1e-5).minimize(loss)
    # define predict function
    predict_function = tf.argmax(logits, 1)
    correct_prediction = tf.equal(predict_function, tf.argmax(labels, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    session.run(init)

    for step in xrange(1000000):
            num_train = train_data.shape[0]
            batch_mask = np.random.choice(num_train, 128)
            train_data_batch = train_data[batch_mask]
            train_labels_batch = label_to_one_hot(train_labels[batch_mask], label_count)

            start_time = time.time()
            train_step.run(feed_dict={images: train_data_batch, labels: train_labels_batch})
            if step % 100 == 0:
                train_accuracy = accuracy.eval(feed_dict={images: train_data_batch, labels: train_labels_batch})
                print("step %d, training accuracy %g" % (step, train_accuracy))
            duration = time.time() - start_time
            print ('duration:%d' % duration)