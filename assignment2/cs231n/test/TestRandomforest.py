"""
 test random forest using iris data set
"""
import pandas as pd
import numpy as np
from cs231n.classifiers.randomforests import *

data = []
data_dir = "E:/github/cs231n/assignment2/cs231n/datasets/iris/"
iris_training = "iris_training.csv"
iris_test = "iris_test.csv"

training_data = pd.read_csv(data_dir + iris_training)
test_data = pd.read_csv(data_dir + iris_test)
test_label = test_data[[4]].values.ravel()
train = list(training_data.iloc[:, 0:5].values)
train = list(train)

test = list(test_data.iloc[:, 0:5].values)
test = list(test)

# training data
classifier = RandomForestsClassifier(n_bootstrapSamples=50)
classifier.fit(train)

resList = []
count = 0
for data in test:
    res = classifier.predict_randomForests(data)
    label = data[4]
    if label == res:
        count += 1
    resList.append(res)
total = len(test_data)
print resList
predict_and_label = np.equal(np.array(test_label), np.array(resList))
correct = [data for data in predict_and_label if data]
num = len(correct)
print ("acc:%0.04f" % (count / (total * 1.0)))
print ("acc:%0.04f" % (num / (total * 1.0)))

# Q: How to save the model random forest
