print(__doc__)
# Author: Krzysztof Sopyla <krzysztofsopyla@gmail.com>
# https://machinethoughts.me
# License: BSD 3 clause


# Standard scientific Python imports

import linearSVM as lsvm
from examples.mnist_helpers import *

# Import datasets, classifiers and performance metrics
from sklearn import datasets, svm, metrics
# fetch original mnist dataset
from sklearn.datasets import load_iris
import datetime as dt


iris = load_iris()

# full dataset classification

samples = 30
X_data = np.concatenate((iris.data[0:samples], iris.data[-1-samples:-1]))
Y = np.concatenate((iris.target[0:samples], iris.target[-1-samples:-1]))

# split data to train and test
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_data, Y, test_size=0.25, random_state=42)


# we set it for testing purposes
X_train=np.array([[ 5.1,  3.5,  1.4,  0.3],
       [ 4.4,  2.9,  1.4,  0.2],
       [ 4.6,  3.4,  1.4,  0.3],
       [ 7.2,  3. ,  5.8,  1.6],
       [ 5. ,  3.6,  1.4,  0.2],
       [ 6.4,  2.8,  5.6,  2.2],
       [ 5.1,  3.8,  1.5,  0.3],
       [ 6.3,  2.7,  4.9,  1.8],
       [ 6.5,  3. ,  5.2,  2. ],
       [ 5. ,  3. ,  1.6,  0.2],
       [ 6.7,  3. ,  5.2,  2.3],
       [ 5.7,  4.4,  1.5,  0.4],
       [ 5.2,  3.5,  1.5,  0.2],
       [ 4.9,  3.1,  1.5,  0.1],
       [ 6. ,  2.2,  5. ,  1.5],
       [ 5. ,  3.4,  1.6,  0.4],
       [ 5.4,  3.9,  1.3,  0.4],
       [ 4.8,  3.4,  1.9,  0.2],
       [ 6.7,  3.3,  5.7,  2.5],
       [ 4.8,  3.4,  1.6,  0.2],
       [ 5.6,  2.8,  4.9,  2. ],
       [ 5.8,  2.7,  5.1,  1.9],
       [ 7.4,  2.8,  6.1,  1.9],
       [ 6.2,  2.8,  4.8,  1.8],
       [ 4.7,  3.2,  1.6,  0.2],
       [ 6.3,  2.8,  5.1,  1.5],
       [ 4.9,  3. ,  1.4,  0.2],
       [ 5.1,  3.7,  1.5,  0.4],
       [ 4.7,  3.2,  1.3,  0.2],
       [ 6.3,  3.4,  5.6,  2.4],
       [ 6.4,  2.8,  5.6,  2.1],
       [ 6.7,  3.3,  5.7,  2.1],
       [ 5.1,  3.3,  1.7,  0.5],
       [ 6. ,  3. ,  4.8,  1.8],
       [ 5.4,  3.7,  1.5,  0.2],
       [ 4.6,  3.6,  1. ,  0.2],
       [ 5.7,  3.8,  1.7,  0.3],
       [ 6.2,  3.4,  5.4,  2.3],
       [ 5.4,  3.4,  1.7,  0.2],
       [ 5. ,  3.4,  1.5,  0.2],
       [ 7.9,  3.8,  6.4,  2. ],
       [ 5.8,  4. ,  1.2,  0.2],
       [ 5.2,  3.4,  1.4,  0.2],
       [ 6.7,  3.1,  5.6,  2.4],
       [ 6.1,  3. ,  4.9,  1.8]])
X_test = X_train

y_train =np.array([0, 0, 0, 2, 0, 2, 0, 2, 2, 0, 2, 0, 0, 0, 2, 0, 0, 0, 2, 0, 2, 2, 2, 2, 0, 2, 0, 0, 0, 2, 2, 2, 0, 2, 0, 0, 0, 2, 0, 0, 2, 0, 0, 2, 2])-1
y_test=y_train


#
# # Create a classifier: a support vector classifier
# classifier = svm.LinearSVC(C=1)
#
#
# print('\n##################\n')
#
# # We learn the digits on train part
# start_time = dt.datetime.now()
# print('Start learning at {}'.format(str(start_time)))
# classifier.fit(X_train, y_train)
# end_time = dt.datetime.now()
# print( 'Stop learning {}'.format(str(end_time)))
# elapsed_time= end_time - start_time
# print('Elapsed learning {}'.format(str(elapsed_time)))
#
#
# # Now predict the value of the test
# expected = y_test
# predicted = classifier.predict(X_test)
#
# acc = np.sum(predicted == expected)/len(expected)
#
# print(classifier.coef_)
#
# print('accuracy={}'.format(acc))
# acc=0
#
# print("Classification report for classifier %s:\n%s\n"
#       % (classifier, metrics.classification_report(expected, predicted)))
#
# cm = metrics.confusion_matrix(expected, predicted)
# print("Confusion matrix:\n%s" % cm)
#
#
# plt.figure()
#
# plot_confusion_matrix(cm)

###
print('\n##################\n')

psvm = lsvm.PrimalSVM(l2reg=0.1)

start_time = dt.datetime.now()
print('Start learning primal,  {}'.format(str(start_time)))
psvm.fit(X_train, y_train)
end_time = dt.datetime.now() 
print('Stop learning {}'.format(str(end_time)))
elapsed_time= end_time - start_time
print('Elapsed learning {}'.format(str(elapsed_time)))



pred, pred_val = psvm.predict(X_test)

acc = np.sum(pred==Y)/len(Y)

print(psvm.w)
print(pred_val)
print('accuracy={}'.format(acc))


