# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Evaluation for CIFAR-10.

Accuracy:
cifar10_train.py achieves 83.0% accuracy after 100K steps (256 epochs
of data) as judged by cifar10_eval.py.

Speed:
On a single Tesla K40, cifar10_train.py processes a single batch of 128 images
in 0.25-0.35 sec (i.e. 350 - 600 images /sec). The model reaches ~86%
accuracy after 100K steps in 8 hours of training time.

Usage:
Please see the tutorial and website for how to download the CIFAR-10
data set, compile the program and train the model.

http://tensorflow.org/tutorials/deep_cnn/
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from datetime import datetime
import math
import time
import numpy as np
import tensorflow as tf
from tensorflow.models.image.cifar10 import cifar10
import cPickle as pickle

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('eval_dir', 'cifar10_eval',
                           """Directory where to write event logs.""")
tf.app.flags.DEFINE_string('eval_data', 'test',
                           """Either 'test' or 'train_eval'.""")
tf.app.flags.DEFINE_string('checkpoint_dir', 'cifar10_train',
                           """Directory where to read model checkpoints.""")
tf.app.flags.DEFINE_integer('eval_interval_secs', 60 * 5,
                            """How often to run the eval.""")
tf.app.flags.DEFINE_integer('num_examples', 300000,
                            """Number of examples to run.""")
tf.app.flags.DEFINE_boolean('run_once', False,
                            """Whether to run eval only once.""")


def eval_once(saver, summary_writer, top_k_op, summary_op):
    """Run Eval once.

  Args:
    saver: Saver.
    summary_writer: Summary writer.
    top_k_op: Top K op.
    summary_op: Summary op.
  """
    with tf.Session() as sess:
        ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            # Restores from checkpoint
            saver.restore(sess, ckpt.model_checkpoint_path)
            # Assuming model_checkpoint_path looks something like:
            #   /my-favorite-path/cifar10_train/model.ckpt-0,
            # extract global_step from it.
            global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
        else:
            print('No checkpoint file found')
            return

        # Start the queue runners.
        coord = tf.train.Coordinator()
        try:
            threads = []
            for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
                threads.extend(qr.create_threads(sess, coord=coord, daemon=True,
                                                 start=True))

            num_iter = int(math.ceil(FLAGS.num_examples / FLAGS.batch_size))
            true_count = 0  # Counts the number of correct predictions.
            total_sample_count = num_iter * FLAGS.batch_size
            step = 0
            while step < num_iter and not coord.should_stop():
                predictions = sess.run([top_k_op])
                true_count += np.sum(predictions)
                step += 1

            # Compute precision @ 1.
            precision = true_count / total_sample_count
            print('%s: precision @ 1 = %.3f' % (datetime.now(), precision))

            summary = tf.Summary()
            summary.ParseFromString(sess.run(summary_op))
            summary.value.add(tag='Precision @ 1', simple_value=precision)
            summary_writer.add_summary(summary, global_step)
        except Exception as e:  # pylint: disable=broad-except
            coord.request_stop(e)

        coord.request_stop()
        coord.join(threads, stop_grace_period_secs=10)


def __load_test_data_batch_with_no_label(filename):
    """ load single batch of cifar """
    with open(filename, 'rb') as f:
        datadict = pickle.load(f)
        X = datadict['data']
        X = X.reshape(10000, 3, 32, 32).transpose(0, 2, 3, 1).astype("float")
        return X


# only load test data for calculation (otherwise MemError)
def loadCIFAR10_Test(ROOT, batch_number):
    Xte = __load_test_data_batch_with_no_label(ROOT + '/test_data_batch_%d' % (batch_number,))
    return Xte


def predict(saver):
    """
  """
    with tf.Session() as sess:
        ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            # Restores from checkpoint
            saver.restore(sess, ckpt.model_checkpoint_path)
            # Assuming model_checkpoint_path looks something like:
            #   /my-favorite-path/cifar10_train/model.ckpt-0,
            # extract global_step from it.
            global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
        else:
            print('No checkpoint file found')
            return

        user_images = tf.placeholder(tf.float32, shape=[None, 32, 32, 3])  # None means whatever size you like

        # x_image = tf.reshape(images, [-1, 32, 32, 3])
        user_logits = cifar10.inference(user_images)

        # define predict function
        predict_function = tf.argmax(user_logits, 1)
        class_list = []
        class_dic = {0: "airplane", 1: "automobile", 2: "bird", 3: "cat", 4: "deer", 5: "dog", 6: "frog", 7: "horse",
                     8: "ship", 9: "truck"}

        BATCH_SIZE = 128
        try:
            predicted_labels = []
            for batch_number in range(1, 31):
                # predicted_labels = []
                test_data = loadCIFAR10_Test('/mnt/hgfs/cs231n/cs231n/assignment2/cs231n/datasets/cifar-10-batches-py',
                                             batch_number)
                test_label = predict_function.eval(feed_dict={user_images: test_data})
                predicted_labels.append(test_label)
                # for i in range(0, test_data.shape[0] // BATCH_SIZE):
                #     test_batch = test_data[i * BATCH_SIZE: (i + 1) * BATCH_SIZE]
                #     predicted_labels[i * BATCH_SIZE: (i + 1) * BATCH_SIZE] = predict_function.eval(feed_dict={images: test_batch})
                #     print("test_batch length: %d" % len(test_batch))
            N = len(predicted_labels)
            for index in xrange(N):
                pre_class = class_dic[predicted_labels[index]]
                class_list.append(pre_class)
            np.savetxt('cifar10_CNN.csv', np.c_[range(1, len(N) + 1), class_list], delimiter=',', header='id,label',
                       comments='', fmt='%s')
        except:
            pass


def evaluate():
    """Eval CIFAR-10 for a number of steps."""
    with tf.Graph().as_default() as g:
        # Get images and labels for CIFAR-10.
        eval_data = FLAGS.eval_data == 'test'
        images, labels = cifar10.inputs(eval_data=eval_data)

        # Build a Graph that computes the logits predictions from the
        # inference model.
        logits = cifar10.inference(images)

        # Calculate predictions.
        top_k_op = tf.nn.in_top_k(logits, labels, 1)

        # define predict function
        predict_function = tf.argmax(logits, 1)

        variable_averages = tf.train.ExponentialMovingAverage(cifar10.MOVING_AVERAGE_DECAY)
        variables_to_restore = variable_averages.variables_to_restore()
        saver = tf.train.Saver(variables_to_restore)
        # Build the summary operation based on the TF collection of Summaries.
        summary_op = tf.merge_all_summaries()
        summary_writer = tf.train.SummaryWriter(FLAGS.eval_dir, g)

        predict(saver)


def main(argv=None):  # pylint: disable=unused-argument

    # Restore the moving average version of the learned variables for eval.
    evaluate()


if __name__ == '__main__':
    tf.app.run()
