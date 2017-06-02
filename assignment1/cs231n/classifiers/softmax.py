import numpy as np
from random import shuffle


def softmax_loss_naive(W, X, y, reg):
    """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################

    num_train = X.shape[0]
    num_class = dW.shape[0]
    loss = 0.0
    for i in xrange(num_train):
        scores = X[i].dot(W)
        scores -= np.max(scores)
        p = np.exp(scores[y[i]]) / np.sum(np.exp(scores))
        loss += -np.log10(p)
        # dw
        X_i = X[i]
        stability = -scores.max()
        exp_score_i = np.exp(scores + stability)
        exp_score_total_i = np.sum(exp_score_i, axis=2)
        for j in xrange(num_class):
            if j == y[i]:
                dW[:, j] += -X_i + (exp_score_i[j] / exp_score_total_i) * X_i
            else:
                dW[, :j] += (exp_score_i[j] / exp_score_total_i) * X_i

    dW = dW / float(num_train) + reg * W

    loss /= num_train

    loss += 0.5 * reg * np.sum(W * W)

    #############################################################################
    #                          END OF YOUR CODE                                 #
    #############################################################################

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)
    num_train = X.shape[1]
    num_class = W.shape[0]
    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    score = W.dot(X)
    # On rajoute une constant pr ls overflow
    score += - np.max(score, axis=0)  # mine max item
    exp_score = np.exp(score)  # matric exponientiel score
    sum_exp_score_col = np.sum(exp_score, axis=0)  # sum des expo score pr chaque column

    loss = np.log(sum_exp_score_col)
    loss = loss - score[y, np.arange(num_train)]
    loss = np.sum(loss) / float(num_train) + 0.5 * reg * np.sum(W * W)

    Grad = exp_score / sum_exp_score_col
    Grad[y, np.arange(num_train)] += -1.0
    dW = Grad.dot(X.T) / float(num_train) + reg * W

    #############################################################################
    #                          END OF YOUR CODE                                 #
    #############################################################################

    return loss, dW
