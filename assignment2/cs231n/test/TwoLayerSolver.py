
from cs231n.test.load_data import *
from cs231n import *

##############################################################################
# TODO: Use a Solver instance to train a TwoLayerNet that achieves at least  #
# 50% accuracy on the validation set.                                        #
##############################################################################
from cs231n.classifiers.fc_net import TwoLayerNet
from cs231n.solver import Solver
from cs231n.test.load_data import *

data = get_CIFAR10_data()  # test inside must have a py filename __init__.py otherwise it can not be referenced
for k, v in data.iteritems():
  print '%s: ' % k, v.shape

model = TwoLayerNet(hidden_dim=100, reg= 8.598929e-03)
solver = Solver(model, data,
                update_rule='sgd',
                optim_config={
                  'learning_rate': 1.207591e-03,
                },
                lr_decay=0.95,
                num_epochs=10, batch_size=100,
                print_every=49000)
solver.train()
##############################################################################
#                             END OF YOUR CODE                               #
##############################################################################