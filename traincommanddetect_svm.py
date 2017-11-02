"""

simple machine learning example with SVM
with PCA dimensionality reduction

from
https://www.kaggle.com/ddmngml/pca-and-svm-on-mnist-dataset

"""

import h5py
import sys
import numpy as np
import numpy

from time import time

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix

import math

infile = sys.argv[1]

f = h5py.File(infile, 'r')
data = f['data'].value
labels = f['labels'].value

X_train = data.reshape((len(labels), -1))
y_train = labels

from sklearn.model_selection import train_test_split
print("splitting into training and test datasets")
X_tr, X_ts, y_tr, y_ts = train_test_split(X_train, y_train, test_size=0.30, random_state=4)

print("dimensionality reduction with PCA")
n_components = 16
t0 = time()
pca = PCA(n_components=n_components, svd_solver='randomized',
          whiten=True).fit(X_train)
print("done in %0.3fs" % (time() - t0))

X_train_pca = pca.transform(X_tr)

print("Training classifier with SVM-RB")
C = 0.1
gamma = 0.05
t0 = time()
clf = SVC(C=C, kernel='rbf', gamma=gamma)
clf = clf.fit(X_train_pca, y_tr)
print("done in %0.3fs" % (time() - t0))

y_pred = clf.predict(pca.transform(X_ts))
score = clf.score(pca.transform(X_ts), y_ts)
print 'score:', score

cnf_matrix = confusion_matrix(y_ts, y_pred)
print 'confusion matrix:'
print cnf_matrix


