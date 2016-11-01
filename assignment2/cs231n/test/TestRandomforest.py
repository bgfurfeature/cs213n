"""
 test random forest
"""

from cs231n.classifiers.randomforests import *

data = []

for i in [0, 1]:
    for j in [0, 1]:
        for k in [0, 1]:
            for m in [0, 1]:
                for v in ["yes", "no"]:
                    data.append([i, j, k, m, v])
# for i in [0, 1]:
#     for j in [0, 1]:
#         for v in ["yes", "no"]:
#             data.append([i, j, 1, 0, v])
#             data.append([i, 0, 1, j, v])

rowRange = range(32)
training_data = [data[i] for i in rowRange[0:22]]
test_data = [data[i] for i in rowRange[22:32]]
print training_data

classifier = RandomForestsClassifier(n_bootstrapSamples=1)
classifier.fit(training_data)
for tree in classifier.list_tree:
    classifier.printTree(tree)
