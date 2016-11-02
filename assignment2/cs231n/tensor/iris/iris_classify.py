from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
import numpy as np

# lod csv file to tensorflow dataset
IRIS_TRAINING_DATA = "iris_training.csv"
IRIS_TEST = "iris_test.csv"
training_set = tf.contrib.learn.dataset.base.load_csv(filename=IRIS_TRAINING_DATA, target_dtype=np.int)
test_set = tf.contrib.learn.dataset.base.load_csv(filename=IRIS_TEST, targrt_dtype=np.int)



# construct a cnn classifier

# fit data

# evaluation

# new samples
