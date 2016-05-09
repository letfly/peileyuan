import numpy as np

from sklearn.datasets import load_digits
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import LabelBinarizer
from neural_network import NeuralNetwork
from sklearn.cross_validation import train_test_split

digits = load_digits()
X = digits.data
y = digits.target
X -= X.min()
X /= X.max()

nn = NeuralNetwork([64, 10, 10], 'logistic')
X_train, X_test, y_train, y_test = train_test_split(X, y)
labels_train = LabelBinarizer().fit_transform(y_train)
labels_test = LabelBinarizer().fit_transform(y_test)
print "start fitting"
nn.fit(X_train, labels_train, epochs=3000)
predictions = []
for i in xrange(X_test.shape[0]):
    o = nn.predict(X_test[i])
    #print 'predict_value',np.argmax(o)
    #print 'true_value',y_test[i];break
    predictions.append(np.argmax(o))
print confusion_matrix(y_test, predictions)
print classification_report(y_test, predictions)

import pylab as pl
#pl.matshow(digits.images[0])
pl.matshow(X_test[0].reshape(8,8))
pl.show()
