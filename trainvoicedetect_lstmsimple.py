"""

Use Keras and a simple Long short-memory neural network to detect voice activity

"""

import sys, os
import numpy
import h5py
import math

# we train on n-1 data sets and evaluate on the remaining one

numpy.random.seed(1)

look_back_seconds = 2
# files are 150000 samples long and cover ~20minutes, so it is
# approximately 140 samples per second
look_back = int(140 * look_back_seconds)
look_back = 1
print 'memory from the last %d frames' % look_back

alldata = []
alllabels = []

for filename in sys.argv[1:]:
	print 'loading...', filename
	f = h5py.File(filename, 'r')
	labels = f['labels'].value
	data = f['data']
	print data.shape, dict(f.attrs)
	print 'attaching memory ...'
	# now, store with memory
	#dataX, dataY = [], []
	# pick out equal numbers for each label
	ioff = numpy.random.choice(numpy.where(labels == 0)[0], replace=False, size=2000)
	ion = numpy.random.choice(numpy.where(labels == 2)[0], replace=False, size=2000)
	print '  selected', ioff.size, ion.size
	for i in numpy.unique(numpy.concatenate((ioff, ion))):
		if i < look_back: continue
		#dataset = (numpy.log(data[i-look_back:i+1,8:] + 1e-3) - numpy.log(1e-3)) / 30.
		#dataset = numpy.log(data[i-look_back:i+1,8:] + 1e-3)
		dataset = data[i-look_back:i+1:,:64]#[::-1][::7*4][::-1]
		#dataset = dataset[:,::4] + dataset[:,1::4] + dataset[:,2::4] + dataset[:,3::4]
		#dataset = dataset[:,::4] + dataset[:,1::4] + dataset[:,2::4] + dataset[:,3::4]
		#img = scipy.misc.imresize(dataset, size=imgshape, mode='F')
		#dataX.append(dataset.flatten())
		#dataY.append(labels[i])
		#print labels[i], numpy.log10(numpy.mean(data[i,8:]))
		alldata.append(dataset.flatten())
		alllabels.append(labels[i])
	print 'attaching memory done.'

X_all = numpy.array(alldata)
Y_all = numpy.array(alllabels)
del alldata
del alllabels

from time import time
from sklearn import preprocessing
from sklearn.preprocessing import quantile_transform
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix

#for preprocessor in ['quantiles', 'normalised-pca', 'log-pca', 'normalised', 'log']:
#for preprocessor in ['normalised-pca', 'log', 'log-pca']:
for preprocessor in ['log-pca', 'log', 'normalised', 'quantiles-pca']:
	print 
	print 'with preprocessor %s:' % preprocessor
	if 'log' in preprocessor:
		X_all2 = numpy.log(X_all + 1e-3)
		X_all2 = preprocessing.scale(X_all2)
	elif 'quantiles' in preprocessor:
		#quantile_transformer = preprocessing.QuantileTransformer(random_state=0)
		#X_all2 = quantile_transformer.fit_transform(X_all)
		X_all2 = preprocessing.quantile_transform(X_all)
	elif 'normalised' in preprocessor:
		X_all2 = preprocessing.scale(X_all)
	else:
		X_all2 = X_all
	print X_all2.min(), X_all2.max(), X_all2.shape, Y_all.shape

	print("splitting into training and test datasets")
	#X_tr, X_ts, y_tr, y_ts = train_test_split(X_all2, Y_all, test_size=0.30, random_state=4)
	i = len(X_all2)*2/3
	X_tr, X_ts, y_tr, y_ts = X_all2[:i], X_all2[i:], Y_all[:i], Y_all[i:]

	Z_train = X_tr
	Z_ts = X_ts

	if 'pca' in preprocessor:
		# now: apply PCA dimensionality reduction
		print("dimensionality reduction with PCA")
		n_components = 16
		t0 = time()
		pca = PCA(n_components=n_components, svd_solver='randomized',
			  whiten=True).fit(X_all2)
		print("done in %0.3fs" % (time() - t0))

		Z_train = pca.transform(X_tr)
		Z_ts = pca.transform(X_ts)

	C = 0.05
	gamma = 0.05
	#for C in 0.01, 0.05, 0.1:
	#	for gamma in 0.01, 0.05, 0.1:
	print("Training classifier with SVM-RB with C=%s gamma=%s" % (C, gamma))
	t0 = time()
	clf = SVC(C=C, kernel='rbf', gamma=gamma)
	clf = clf.fit(Z_train, y_tr)
	print("done in %0.3fs" % (time() - t0))

	y_pred = clf.predict(Z_ts)
	score = clf.score(Z_ts, y_ts)
	print 'score:', score

	cnf_matrix = confusion_matrix(y_ts, y_pred)
	print 'confusion matrix:'
	print cnf_matrix



