# coding=utf-8
import numpy as np
from cs231n.layers import *
from cs231n.fast_layers import *
from cs231n.layer_utils import *

"""
Digit Recognizer in Python using Convolutional Neural Nets¶
"""


# user define layer， whatever you like ，just give me higher-accuracy
class conv_relu_max_pool_affine_relu_affineNet(object):
    """
  A three-layer convolutional network with the following architecture:

  conv - relu - 2x2 max pool - affine - relu - affine - softmax

  The network operates on minibatches of data that have shape (N, C, H, W)
  consisting of N images, each with height H and width W and with C input
  channels.
  """

    def __init__(self, input_dim=(1, 28, 28), num_filters=7, filter_size=3,
                 hidden_dim=100, num_classes=10, weight_scale=1e-3, reg=0.0,
                 dtype=np.float32):
        """
    Initialize a new network.

    Inputs:
    - input_dim: Tuple (C, H, W) giving size of input data
    - num_filters: Number of filters to use in the convolutional layer
    - filter_size: Size of filters to use in the convolutional layer
    - hidden_dim: Number of units to use in the fully-connected hidden layer
    - num_classes: Number of scores to produce from the final affine layer.
    - weight_scale: Scalar giving standard deviation for random initialization
      of weights.
    - reg: Scalar giving L2 regularization strength
    - dtype: numpy datatype to use for computation.
    """
        self.params = {}
        self.reg = reg
        self.dtype = dtype

        ############################################################################
        # TODO: Initialize weights and biases for the three-layer convolutional    #
        # network. Weights should be initialized from a Gaussian with standard     #
        # deviation equal to weight_scale; biases should be initialized to zero.   #
        # All weights and biases should be stored in the dictionary self.params.   #
        # Store weights and biases for the convolutional layer using the keys 'W1' #
        # and 'b1'; use keys 'W2' and 'b2' for the weights and biases of the       #
        # hidden affine layer, and keys 'W3' and 'b3' for the weights and biases   #
        # of the output affine layer.                                              #
        ############################################################################
        C, H, W = input_dim
        self.params['W1'] = np.random.normal(0, weight_scale, (num_filters, C, filter_size, filter_size))
        self.params['b1'] = np.zeros(num_filters)
        self.params['W2'] = np.random.normal(0, weight_scale, (num_filters * H / 2 * W / 2, hidden_dim))
        self.params['b2'] = np.zeros(hidden_dim)
        self.params['W3'] = np.random.normal(0, weight_scale, (hidden_dim, num_classes))
        self.params['b3'] = np.zeros(num_classes)
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        for k, v in self.params.iteritems():
            self.params[k] = v.astype(dtype)

    def loss(self, X, y=None):
        """
    Evaluate loss and gradient for the three-layer convolutional network.

    Input / output: Same API as TwoLayerNet in fc_net.py.
    """
        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']
        W3, b3 = self.params['W3'], self.params['b3']

        # pass conv_param to the forward pass for the convolutional layer
        filter_size = W1.shape[2]
        conv_param = {'stride': 1, 'pad': (filter_size - 1) / 2}

        # pass pool_param to the forward pass for the max-pooling layer
        pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the three-layer convolutional net,  #
        # computing the class scores for X and storing them in the scores          #
        # variable.                                                                #
        ############################################################################

        # conv_out, conv_cache = conv_forward_im2col(X, W1, b1, conv_param)  # need to set right env
        conv_out, conv_cache = conv_forward_fast(X, W1, b1, conv_param)
        relu1_out, relu1_cache = relu_forward(conv_out)
        pool_out, pool_cache = max_pool_forward_fast(relu1_out, pool_param)
        affine_relu_out, affine_relu_cache = affine_relu_forward(pool_out, W2, b2)
        affine2_out, affine2_cache = affine_forward(affine_relu_out, W3, b3)
        scores = affine2_out

        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        if y is None:
            return scores

        loss, grads = 0, {}
        ############################################################################
        # TODO: Implement the backward pass for the three-layer convolutional net, #
        # storing the loss and gradients in the loss and grads variables. Compute  #
        # data loss using softmax, and make sure that grads[k] holds the gradients #
        # for self.params[k]. Don't forget to add L2 regularization!               #
        ############################################################################
        loss, dscores = softmax_loss(scores, y)
        loss += 0.5 * self.reg * (
            np.sum(self.params['W1'] * self.params['W1']) + np.sum(self.params['W2'] * self.params['W2']) + np.sum(
                self.params['W3'] * self.params['W3']))

        affine2_dx, affine2_dw, affine2_db = affine_backward(dscores, affine2_cache)
        grads['W3'] = affine2_dw + self.reg * self.params['W3']
        grads['b3'] = affine2_db

        affine1_dx, affine1_dw, affine1_db = affine_relu_backward(affine2_dx, affine_relu_cache)
        grads['W2'] = affine1_dw + self.reg * self.params['W2']
        grads['b2'] = affine1_db

        pool_dx = max_pool_backward_fast(affine1_dx, pool_cache)
        relu_dx = relu_backward(pool_dx, relu1_cache)
        # conv_dx, conv_dw, conv_db = conv_backward_im2col(relu_dx, conv_cache)  # need to set right env
        conv_dx, conv_dw, conv_db = conv_backward_naive(relu_dx, conv_cache)
        grads['W1'] = conv_dw + self.reg * self.params['W1']
        grads['b1'] = conv_db

        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads

    ########################################################################################################################
    # LeNet try : two convlayer didn't work well, there are something wrong that i am not figured out                      #
    # firstConv -> 28 x 28 x1 with filter 5 x 5 x 32, stride 1, pad 2 transformed to 28 x 28 x 32                          #
    # maxpool ->  28 x 28 x 32 with pool size 2 x 2, stride 2 transformed to 14 x 14 x 32                                  #
    # secondConv -> 14 x 14 x 32 with  filter 5 x 5 x 64, stride 1, pad 2 transformed to 14 x 14 x 64                        #
    # maxpool -> 14 x 14 x 64 with pool size 2 x 2, stride 2 transformed to 7 x 7 x 64                                       #
    # fc_1 -> neuron number 1024
    # droup out 0.5                                                                                          #
    # fc_2 -> neuron number 84 # abandon                                                                                            #
    # out -> classes 10                                                                                                    #
    #


#######################################################################################################################
# user define layer， whatever you like ，just give me higher-accuracy
class LeNet(object):
    """
  A three-layer convolutional network with the following architecture:

  conv - relu(5x5x6) - 2x2 max pool - conv - relu(5x5x16) - 2x2 max pool - affine(120) - relu - affine(84) - affine(10) - softmax

  The network operates on minibatches of data that have shape (N, C, H, W)
  consisting of N images, each with height H and width W and with C input
  channels.
  """

    def __init__(self, input_dim=(1, 28, 28), num_filters1=32, num_filters2=64, filter_size=5,
                 hidden_dim1=1024, hidden_dim2=84, num_classes=10, weight_scale=1e-3, reg=0.0,
                 dtype=np.float32):
        """
    Initialize a new network.

    Inputs:
    - input_dim: Tuple (C, H, W) giving size of input data
    - num_filters: Number of filters to use in the convolutional layer
    - filter_size: Size of filters to use in the convolutional layer
    - hidden_dim: Number of units to use in the fully-connected hidden layer
    - num_classes: Number of scores to produce from the final affine layer.
    - weight_scale: Scalar giving standard deviation for random initialization
      of weights.
    - reg: Scalar giving L2 regularization strength
    - dtype: numpy datatype to use for computation.
    """
        self.params = {}
        self.reg = reg
        self.dtype = dtype

        ############################################################################
        # TODO: Initialize weights and biases for the three-layer convolutional    #
        # network. Weights should be initialized from a Gaussian with standard     #
        # deviation equal to weight_scale; biases should be initialized to zero.   #
        # All weights and biases should be stored in the dictionary self.params.   #
        # Store weights and biases for the convolutional layer using the keys 'W1' #
        # and 'b1'; use keys 'W2' and 'b2' for the weights and biases of the       #
        # hidden affine layer, and keys 'W3' and 'b3' for the weights and biases   #
        # of the output affine layer.                                              #
        ############################################################################
        C, H, W = input_dim

        self.params['W1'] = np.random.normal(0, weight_scale, (num_filters1, C, filter_size, filter_size))
        self.params['b1'] = np.zeros(num_filters1)

        self.params['W1_2'] = np.random.normal(0, weight_scale, (num_filters2, num_filters1, filter_size, filter_size))
        self.params['b1_2'] = np.zeros(num_filters2)
        self.params['W2_2'] = np.random.normal(0, weight_scale, (num_filters2 * H / 2 * W / 2, hidden_dim1))
        self.params['b2_2'] = np.zeros(hidden_dim1)  # for double conv layer setting

        self.params['W2'] = np.random.normal(0, weight_scale, (num_filters2 * H / 2 * W / 2, hidden_dim1))
        self.params['b2'] = np.zeros(hidden_dim1)

        self.params['W3'] = np.random.normal(0, weight_scale, (hidden_dim1, hidden_dim2))
        self.params['b3'] = np.zeros(hidden_dim2)

        self.params['W4'] = np.random.normal(0, weight_scale, (hidden_dim2, num_classes))
        self.params['b4'] = np.zeros(num_classes)

        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        for k, v in self.params.iteritems():
            self.params[k] = v.astype(dtype)

    def loss(self, X, y=None):
        """
    Evaluate loss and gradient for the three-layer convolutional network.

    Input / output: Same API as TwoLayerNet in fc_net.py.
    """
        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']
        W3, b3 = self.params['W3'], self.params['b3']
        W4, b4 = self.params['W4'], self.params['b4']

        # pass conv_param to the forward pass for the convolutional layer
        filter_size = W1.shape[2]
        conv_param1 = {'stride': 1, 'pad': (filter_size - 1) / 2}

        # pass pool_param to the forward pass for the max-pooling layer
        pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

        scores = None

        ############################################################################
        # TODO: Implement the forward pass for the three-layer convolutional net,  #
        # computing the class scores for X and storing them in the scores          #
        # variable.                                                                #
        ############################################################################

        # conv_out, conv_cache = conv_forward_im2col(X, W1, b1, conv_param)  # need to set right env
        conv_out, conv_cache = conv_forward_fast(X, W1, b1, conv_param1)
        relu1_out, relu1_cache = relu_forward(conv_out)
        pool_out, pool_cache = max_pool_forward_fast(relu1_out, pool_param)  # can not use fast method because of back_ward method, backward method between two convLayer can not be execute correctly

        affine_relu_out, affine_relu_cache = affine_relu_forward(pool_out, W2, b2)
        affine_relu_out2, affine_relu_cache2 = affine_relu_forward(affine_relu_out, W3, b3)

        affine2_out, affine2_cache = affine_forward(affine_relu_out2, W4, b4)

        scores = affine2_out

        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        if y is None:
            return scores

        loss, grads = 0, {}
        ############################################################################
        # TODO: Implement the backward pass for the three-layer convolutional net, #
        # storing the loss and gradients in the loss and grads variables. Compute  #
        # data loss using softmax, and make sure that grads[k] holds the gradients #
        # for self.params[k]. Don't forget to add L2 regularization!               #
        ############################################################################

        loss, dscores = softmax_loss(scores, y)

        loss += 0.5 * self.reg * (np.sum(
            self.params['W1'] * self.params['W1']) + np.sum(
            self.params['W2'] * self.params['W2']) + np.sum(
            self.params['W3'] * self.params['W3']) + np.sum(
            self.params['W4'] * self.params['W4']))  # + np.sum(self.params['W1_2'] * self.params['W1_2']) )

        affine2_dx, affine2_dw, affine2_db = affine_backward(dscores, affine2_cache)
        grads['W4'] = affine2_dw + self.reg * self.params['W4']
        grads['b4'] = affine2_db

        affine3_dx, affine3_dw, affine3_db = affine_relu_backward(affine2_dx, affine_relu_cache2)

        grads['W3'] = affine3_dw + self.reg * self.params['W3']
        grads['b3'] = affine3_db

        affine1_dx, affine1_dw, affine1_db = affine_relu_backward(affine3_dx, affine_relu_cache)
        grads['W2'] = affine1_dw + self.reg * self.params['W2']
        grads['b2'] = affine1_db

        pool2_dx = max_pool_backward_fast(affine1_dx, pool_cache)
        relu2_dx = relu_backward(pool2_dx, relu1_cache)
        # conv_dx, conv_dw, conv_db = conv_backward_im2col(relu_dx, conv_cache)  # need to set right env
        conv2_dx, conv2_dw, conv2_db = conv_backward_naive(relu2_dx, conv_cache)
        grads['W1'] = conv2_dw + self.reg * self.params['W1']
        grads['b1'] = conv2_db

        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads

    def refactor_loss(self, X, y=None):
        """
         refactor fast_ward and backward method for simplify
         how to know where is wrong with the network? how to check the dead neurons ?
         Evaluate loss and gradient for the three-layer convolutional network.
         Input / output: Same API as TwoLayerNet in fc_net.py.

         :param y:
        """
        W1, b1 = self.params['W1'], self.params['b1']
        W1_2, b1_2 = self.params['W1_2'], self.params['b1_2']
        W2_2, b2_2 = self.params['W2_2'], self.params['b2_2']
        W3, b3 = self.params['W3'], self.params['b3']
        W4, b4 = self.params['W4'], self.params['b4']

        # pass conv_param to the forward pass for the convolutional layer
        filter_size = W1.shape[2]
        conv_param1 = {'stride': 1, 'pad': (filter_size - 1) / 2}
        # pass pool_param to the forward pass for the max-pooling layer
        pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the three-layer convolutional net,  #
        # computing the class scores for X and storing them in the scores          #
        # variable.                                                                #
        ############################################################################

        out1, cache1 = conv_relu_pool_forward(X, W1, b1, conv_param1, pool_param)
        # print out1.shape

        out2, cache2 = conv_relu_pool_forward(out1, W1_2, b1_2, conv_param1, pool_param)

        # print out2.shape

        affine_relu_out, affine_relu_cache = affine_relu_forward(out2, W2_2, W2_2)

        # print affine_relu_out.shape

        affine_relu_out2, affine_relu_cache2 = affine_relu_forward(affine_relu_out, W3, b3)

        # print affine_relu_out2.shape

        affine2_out, affine2_cache = affine_forward(affine_relu_out2, W4, b4)

        # print affine2_out.shape

        # print "back_ward ###############################################"

        scores = affine2_out

        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        if y is None:
            return scores

        loss, grads = 0, {}
        ############################################################################
        # TODO: Implement the backward pass for the three-layer convolutional net, #
        # storing the loss and gradients in the loss and grads variables. Compute  #
        # data loss using softmax, and make sure that grads[k] holds the gradients #
        # for self.params[k]. Don't forget to add L2 regularization!               #
        ############################################################################

        loss, dscores = softmax_loss(scores, y)

        loss += 0.5 * self.reg * (np.sum(
            self.params['W1'] * self.params['W1']) + np.sum(
            self.params['W2_2'] * self.params['W2_2']) + np.sum(
            self.params['W3'] * self.params['W3']) + np.sum(
            self.params['W4'] * self.params['W4']) + np.sum(
            self.params['W1_2'] * self.params['W1_2']))

        affine2_dx, affine2_dw, affine2_db = affine_backward(dscores, affine2_cache)

        grads['W4'] = affine2_dw + self.reg * self.params['W4']
        grads['b4'] = affine2_db

        # print affine2_dx.shape

        affine3_dx, affine3_dw, affine3_db = affine_relu_backward(affine2_dx, affine_relu_cache2)

        # print affine3_dx.shape

        grads['W3'] = affine3_dw + self.reg * self.params['W3']
        grads['b3'] = affine3_db

        affine1_dx, affine1_dw, affine1_db = affine_relu_backward(affine3_dx, affine_relu_cache)
        grads['W2_2'] = affine1_dw + self.reg * self.params['W2_2']
        grads['W2_2'] = affine1_db

        # print affine1_dx.shape

        out1_dout, conv_dw, conv_db = conv_relu_pool_backward_naive(affine1_dx, cache2)
        grads['W1_2'] = conv_dw + self.reg * self.params['W1_2']
        grads['b1_2'] = conv_db

        out_dout, w_dw, d_db = conv_relu_pool_backward_naive(out1_dout, cache1)

        grads['W1'] = w_dw + self.reg * self.params['W1']
        grads['b1'] = d_db

        # print out_dout.shape

        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads
