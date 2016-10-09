"""
custom network structures
"""

from cs231n.test.load_data import *
from cs231n import *
##############################################################################
# TODO: Use a Solver instance to train a TwoLayerNet that achieves at least  #
# 50% accuracy on the validation set.                                        #
##############################################################################
from cs231n.classifiers.fc_net import *
from cs231n.solver import Solver
from cs231n.test.load_data import *

# LOAD DATA
data = get_CIFAR10_data()  # test inside must have a py filename __init__.py otherwise it can not be referenced
for k, v in data.iteritems():
    print '%s: ' % k, v.shape

X_test = data['X_test']
y_test = data['y_test']

X_val = data['X_val']
y_val = data['y_val']

# 2-layers network

# model = TwoLayerNet(hidden_dim=100, reg= 8.598929e-03)
# solver = Solver(model, data,
#                 update_rule='sgd',
#                 optim_config={
#                   'learning_rate': 1.207591e-03,
#                 },
#                 lr_decay=0.95,
#                 num_epochs=10, batch_size=100,
#                 print_every=49000)
# solver.train()

# 5-layers network

lr = 3.113669e-04  # [10] #[2.379994e-04] # 10**np.random.uniform(-7,1,20)
ws = 2.461858e-02
model = FullyConnectedNet([100, 100, 100, 100],
                          weight_scale=ws, dtype=np.float64, use_batchnorm=False, reg=1e-2)
solver = Solver(model, data,
                print_every=100, num_epochs=10, batch_size=25,
                update_rule='adam',
                optim_config={
                    'learning_rate': lr,
                },
                lr_decay=0.9,
                verbose=True
                )

solver.train()

best_model = model

# Test you model

y_test_pred = np.argmax(best_model.loss(X_test), axis=1)
y_val_pred = np.argmax(best_model.loss(X_val), axis=1)
print 'Validation set accuracy: ', (y_val_pred == y_val).mean()
print 'Test set accuracy: ', (y_test_pred == y_test).mean()
##############################################################################
#                             END OF YOUR CODE                               #
##############################################################################
