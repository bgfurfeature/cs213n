# coding=utf-8
# pylint: disable=missing-docstring
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import time
import numpy as np
import tensorflow as tf
import numpy as np
import cPickle as pickle
import os


class cifarTrain(object):
    def __init__(self):
        self.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 10000
        self.desc = "This is a train for specific traning data CIFAR-10!!"
        # Constants describing the training process.
        self.MOVING_AVERAGE_DECAY = 0.9999  # The decay to use for the moving average.
        self.NUM_EPOCHS_PER_DECAY = 350.0  # Epochs after which learning rate decays.
        self.LEARNING_RATE_DECAY_FACTOR = 0.1  # Learning rate decay factor.
        self.INITIAL_LEARNING_RATE = 0.1  # Initial learning rate.
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
            tf.scalar_summary(l.op.name + ' (raw)', l)
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
    def __init__(self, flag='float32', batch_size=50, NUM_CLASSES=10):
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

        initial = tf.truncated_normal_initializer(stddev=stddev, dtype=dtype)
        var = tf.get_variable(name=name, shape=shape, dtype=dtype, initializer=initial)

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
            reshape = tf.reshape(pool2, [-1, 8 * 8 * 64])
            # dim = reshape.get_shape()[1]
            weights = self._variable_with_weight_decay('weights', shape=[8 * 8 * 64, 384], stddev=0.04, wd=0.004)
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
            biases = self.bias('biases', [self.NUM_CLASSES], value=0.0)
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
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits, labels, name='cross_entropy_per_example')
        cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
        tf.add_to_collection('losses', cross_entropy_mean)  # data loss

        # The total loss is defined as the cross entropy loss plus all of the weight
        # decay terms (L2 loss).
        # data loss + weight loss
        return tf.add_n(tf.get_collection('losses'), name='total_loss')


def __load_CIFAR_batch(filename):
    """ load single batch of cifar """
    with open(filename, 'rb') as f:
        datadict = pickle.load(f)
        X = datadict['data']
        Y = datadict['labels']
        X = X.reshape(10000, 3, 32, 32).transpose(0, 2, 3, 1).astype("float")
        Y = np.array(Y)
        return X, Y


def __load_test_data_batch_with_no_label(filename):
    """ load single batch of cifar """
    with open(filename, 'rb') as f:
        datadict = pickle.load(f)
        X = datadict['data']
        X = X.reshape(10000, 3, 32, 32).transpose(0, 2, 3, 1).astype("float")
        return X


def __load_CIFAR10(ROOT):
    """ load all of cifar """
    xs = []
    xs1 = []
    ys = []
    for b in range(1, 6):
        # f = os.path.join(ROOT, 'data_batch_%d' % (b,))
        X, Y = __load_CIFAR_batch(ROOT + "/" + 'data_batch_%d' % (b,))
        xs.append(X)
        ys.append(Y)
    Xtr = np.concatenate(xs)
    Ytr = np.concatenate(ys)
    del X, Y, xs, ys
    Xval, Yval = __load_CIFAR_batch(ROOT + '/test_batch')

    # for a in range(1, 31):
    #     X1 = __load_test_data_batch_with_no_label(ROOT + '/test_data_batch_%d' % (a,))
    #     xs1.append(X1)
    # Xte = np.concatenate(xs1)
    # del X1, xs1
    return Xtr, Ytr, Xval, Yval  # , Xte


# only load test data for calculation (otherwise MemError)
def loadCIFAR10_Test(ROOT, batch_number):
    Xte = __load_test_data_batch_with_no_label(ROOT + '/test_data_batch_%d' % (batch_number,))
    return Xte

def get_CIFAR10_data(file_name, num_training=50000, num_validation=10000, num_test=10000):
    """
    Load the CIFAR-10 dataset from disk and perform preprocessing to prepare
    it for classifiers. These are the same steps as we used for the SVM, but
    condensed to a single function.
    """
    # Load the raw CIFAR-10 data
    cifar10_dir = file_name
    X_train, y_train, X_val, y_val = __load_CIFAR10(cifar10_dir)

    # Subsample the data
    # print X_train.shape
    # Normalize the data: subtract the mean image, same mean pls
    mean_image = np.mean(X_train, axis=0)
    X_train -= mean_image
    X_val -= mean_image

    return {
        'X_train': X_train, 'y_train': y_train,
        'X_val': X_val, 'y_val': y_val,
        'mean': mean_image
    }


# 0 => [1 0 0 0 0 0 0 0 0 0]
# 1 => [0 1 0 0 0 0 0 0 0 0]
# ...
# 9 => [0 0 0 0 0 0 0 0 0 1]
def label_to_one_hot(labels, num_classes):
    num_batch = labels.shape[0]
    index_offset = np.arange(
        num_batch) * num_classes  # each row has num_classes item，so for each row the offset is row_number * num_classes
    label_flat_hot = np.zeros((num_batch, num_classes))
    label_flat_hot.flat[index_offset + labels.ravel()] = 1
    return label_flat_hot


if __name__ == '__main__':

    cifarModel = cifarModel()
    global_step = tf.Variable(0, trainable=False)
    # load_data
    data = get_CIFAR10_data('/mnt/hgfs/cs231n/cs231n/assignment2/cs231n/datasets/cifar-10-batches-py')

    train_data = data['X_train']
    train_labels = data['y_train']
    val_data = data['X_val']
    val_label = data['y_val']
    mean_image = data['mean']

    print("train_data length: %d" % len(train_data))
    print("val_data length: %d" % len(val_data))
    # print("test_data length: %d" % len(test_data))

    class_list = []
    # class_list_2 = []
    class_dic = {0: "airplane", 1: "automobile", 2: "bird", 3: "cat", 4: "deer", 5: "dog", 6: "frog", 7: "horse",
                 8: "ship", 9: "truck"}
    # N = test_data.shape[0]
    # for index in xrange(N):
    #         pre_class_2 = class_dic[test_label[index]]
    #         class_list_2.append(pre_class_2)
    #
    # np.savetxt('cifar10_CNN_no_train_get.csv', np.c_[range(1, len(test_data) + 1), class_list_2], delimiter=',',
    #            header='id,label', comments='', fmt='%s')

    label_count = np.unique(train_labels).shape[0]

    images = tf.placeholder(tf.float32, shape=[None, 32, 32, 3])  # None means whatever size you like
    labels = tf.placeholder(tf.float32, shape=[None, cifarModel.NUM_CLASSES])

    # graph
    session = tf.InteractiveSession()

    # x_image = tf.reshape(images, [-1, 32, 32, 3])
    logits = cifarModel.process(images)

    # Calculate loss.
    loss_ = cifarModel.loss(logits, labels)
    train_step = tf.train.AdadeltaOptimizer(1e-5).minimize(loss_)
    # define predict function
    predict_function = tf.argmax(logits, 1)
    correct_prediction = tf.equal(predict_function, tf.argmax(labels, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    # Create a saver.
    saver = tf.train.Saver(tf.all_variables())
    session.run(tf.initialize_all_variables())

    BATCH_SIZE = 128
    max_steps = 1000000
    train_model_dir = '/mnt/hgfs/cs231n/cs231n/assignment2/cs231n/datasets/cifar-10-batches-py/checkpoint_dir'
    for step in xrange(max_steps):
        num_train = train_data.shape[0]
        batch_mask = np.random.choice(num_train, BATCH_SIZE)
        train_data_batch = train_data[batch_mask]
        train_labels_batch = label_to_one_hot(train_labels[batch_mask], label_count)

        start_time = time.time()
        train_step.run(feed_dict={images: train_data_batch, labels: train_labels_batch})
        if step % 100 == 0:
            duration = time.time() - start_time
            print('duration:%d s' % duration)
            train_accuracy = accuracy.eval(feed_dict={images: train_data_batch, labels: train_labels_batch})
            print("step %d, training accuracy %g" % (step, train_accuracy))

        # Save the model checkpoint periodically.
        if step % 1000 == 0 or (step + 1) == max_steps:
            checkpoint_path = os.path.join(train_model_dir, 'model.ckpt')
            saver.save(session, checkpoint_path, global_step=step)

    print("val accuracy %g" % accuracy.eval(
        feed_dict={images: val_data, labels: label_to_one_hot(val_label, label_count)}))

    # using batches is more resource efficient
    # predicted_lables = np.zeros(test_data.shape[0])
    # for i in range(0, test_data.shape[0] // BATCH_SIZE):
    #     test_batch = test_data[i * BATCH_SIZE: (i + 1) * BATCH_SIZE]
    #     predicted_lables[i * BATCH_SIZE: (i + 1) * BATCH_SIZE] = predict_function.eval(feed_dict={images: test_batch})
    try:
        predicted_labels = []
        for batch_number in range(1, 31):
            # predicted_labels = []
            test_batch = loadCIFAR10_Test('/mnt/hgfs/cs231n/cs231n/assignment2/cs231n/datasets/cifar-10-batches-py', batch_number)
            test_batch -= mean_image
            print("test_batch length: %d" % len(test_batch))
            test_label = predict_function.eval(feed_dict={images: test_batch})
            predicted_labels.append(test_label)
        N = len(predicted_labels)
        for index in xrange(N):
            pre_class = class_dic[predicted_labels[index]]
            class_list.append(pre_class)
        # for index in xrange(N):
        #     pre_class_2 = class_dic[val_label[index]]
        #     class_list_2.append(pre_class_2)
        # save results
        np.savetxt('cifar10_CNN.csv', np.c_[range(1, len(N) + 1), class_list], delimiter=',', header='id,label', comments='', fmt='%s')
    except:
        pass


