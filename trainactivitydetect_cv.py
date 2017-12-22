"""

Detect voice activity

Here we test and train multiple methods with cross-validation.

Long short term memory neural networks
SVM
Random Forest

Notes: RandomForest and MLP10 do very well (95% correctness) with log preprocessor
Worse if only current sample is considered (90%)
Worse if only current sample and 3s median is considered (91%)

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
#look_back = 0
print 'memory from the last %d frames' % look_back

alldata = []
alllabels = []

for filename in sys.argv[1:]:
	print 'loading...', filename
	f = h5py.File(filename, 'r')
	labels = f['labels'].value
	data = f['data']
	print '  ', data.shape, dict(f.attrs)
	print '  attaching memory ...'
	# now, store with memory
	#dataX, dataY = [], []
	# pick out equal numbers for each label
	ioff = numpy.random.choice(numpy.where(labels == 0)[0], replace=False, size=200)
	ion = numpy.random.choice(numpy.where(labels == 2)[0], replace=False, size=200)
	print '    selected', ioff.size, ion.size
	for i in numpy.unique(numpy.concatenate((ioff, ion))):
		if i < look_back: continue
		#dataset = (numpy.log(data[i-look_back:i+1,8:] + 1e-3) - numpy.log(1e-3)) / 30.
		#dataset = numpy.log(data[i-look_back:i+1,8:] + 1e-3)
		dataset = data[i-look_back:i+1:,:64][::-1][::7][::-1]
		#dataset = data[i,:]
		#dataset2 = numpy.median(data[i-look_back:i+1:,:], axis=0)
		#assert dataset.shape == dataset2.shape, (dataset.shape, dataset2.shape)
		#dataset = numpy.vstack((dataset, dataset2))
		#dataset = dataset[:,::4] + dataset[:,1::4] + dataset[:,2::4] + dataset[:,3::4]
		#dataset = dataset[:,::4] + dataset[:,1::4] + dataset[:,2::4] + dataset[:,3::4]
		#img = scipy.misc.imresize(dataset, size=imgshape, mode='F')
		#dataX.append(dataset.flatten())
		#dataY.append(labels[i])
		#print labels[i], numpy.log10(numpy.mean(data[i,8:]))
		alldata.append(dataset.flatten())
		alllabels.append(labels[i]//2)
	print '  attaching memory done.'

X_all = numpy.array(alldata)
Y_all = numpy.array(alllabels)
del alldata
del alllabels
print 'data:', X_all.shape, Y_all.shape

from time import time
from sklearn import preprocessing
from sklearn.preprocessing import quantile_transform
from sklearn.decomposition import PCA, KernelPCA
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, roc_curve
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import fbeta_score, make_scorer
import matplotlib.pyplot as plt
ftwo_scorer = make_scorer(fbeta_score, beta=0.2)

def train_and_evaluate(name, clf):
	print
	sys.stdout.write('running %s ...\r' % name)
	sys.stdout.flush()
	t0 = time()
	q = cross_val_score(clf, X_all2, Y_all, cv=5, scoring=ftwo_scorer)
	print '%2.2f %s (training speed: %.1fs)' % (q.mean(), name, time() - t0)

	i = len(X_all2)*2/3
	X_tr, X_ts, y_tr, y_ts = Z_all[:i], Z_all[i:], Y_all[:i], Y_all[i:]
	clf = clf.fit(X_tr, y_tr)
	t0 = time()
	for i in range(10):
		y_pred = clf.predict(X_ts)
	y_scores = clf.predict_proba(X_ts)
	print 'confusion matrix: (eval speed: %.2fs)' % (time() - t0)
	cnf_matrix = confusion_matrix(y_ts, y_pred)
	print cnf_matrix
	print 'ROC curve plot...'
	#print y_scores[:,0] + y_scores[:,1], (y_scores[:,0] + y_scores[:,1]).min(), (y_scores[:,0] + y_scores[:,1]).max()
	fpr, tpr, thresholds = roc_curve(y_ts, y_scores[:,1])
	plt.title(name)
	#print fpr, tpr, thresholds
	print '5% FPR: at threshold', thresholds[fpr < 0.05][-1], 'with efficiency', tpr[fpr < 0.05][-1]*100, '%'
	print '1% FPR: at threshold', thresholds[fpr < 0.01][-1], 'with efficiency', tpr[fpr < 0.01][-1]*100, '%'
	plt.plot(fpr, tpr, '-', color='r')
	plt.plot([0,1], [0,1], ':', color='k')
	plt.xlabel('False positive rate')
	plt.ylabel('True positive rate')
	plt.savefig('trainactivitydetect_scores_%s.pdf' % name, bbox_inches='tight')
	plt.close()
	

#for preprocessor in ['quantiles', 'normalised-pca', 'log-pca', 'normalised', 'log']:
#for preprocessor in ['normalised-pca', 'log', 'log-pca']:
for preprocessor in ['log', 'logscale']: #, 'log-pca', 'normalised']: #, 'quantiles-pca']:
	print 
	print 'with preprocessor %s:' % preprocessor
	if 'log' in preprocessor:
		X_all2 = (numpy.log(X_all + 1e-3) - numpy.log(1e-3)) / 30.
	elif 'logscale' in preprocessor:
		X_all2 = (numpy.log(X_all + 1e-3) - numpy.log(1e-3)) / 30.
		X_all2 = preprocessing.scale(X_all2)
	elif 'quantiles' in preprocessor:
		#quantile_transformer = preprocessing.QuantileTransformer(random_state=0)
		#X_all2 = quantile_transformer.fit_transform(X_all)
		X_all2 = preprocessing.quantile_transform(X_all)
	elif 'normalised' in preprocessor:
		X_all2 = preprocessing.scale(X_all / X_all.max())
	else:
		X_all2 = X_all
	#print X_all2.min(), X_all2.max(), X_all2.shape, Y_all.shape

	Z_all = X_all2

	if 'kpca' in preprocessor:
		# now: apply PCA dimensionality reduction
		print "  dimensionality reduction with KernelPCA"
		n_components = 16
		t0 = time()
		pca = KernelPCA(n_components=n_components, kernel='rbf').fit(X_all2)
		print "    done in %0.3fs" % (time() - t0)

		Z_all = pca.transform(X_all2)
	elif 'pca' in preprocessor:
		# now: apply PCA dimensionality reduction
		print "  dimensionality reduction with PCA"
		n_components = 16
		t0 = time()
		pca = PCA(n_components=n_components, svd_solver='randomized',
			  whiten=True).fit(X_all2)
		print "    done in %0.3fs" % (time() - t0)

		Z_all = pca.transform(X_all2)

	if 'log' == preprocessor:
		with h5py.File('voicedetect-training.hdf5', 'w') as f:
			f.create_dataset('x7', data=Z_all, compression='gzip', shuffle=True)
			f.create_dataset('y', data=Y_all, compression='gzip', shuffle=True)
		
		#train_and_evaluate('DecisionTree', clf = DecisionTreeClassifier())
	
		train_and_evaluate('RandomForest1', clf = RandomForestClassifier(n_estimators=1))
		train_and_evaluate('RandomForest4', clf = RandomForestClassifier(n_estimators=4))
		train_and_evaluate('RandomForest10', clf = RandomForestClassifier(n_estimators=10))
		train_and_evaluate('RandomForest40', clf = RandomForestClassifier(n_estimators=40))
		#train_and_evaluate('RandomForest100', clf = RandomForestClassifier(n_estimators=100))
	
		train_and_evaluate('AdaBoost', clf = AdaBoostClassifier(n_estimators=40))
	
		train_and_evaluate('GradientBoosting', clf = GradientBoostingClassifier(n_estimators=40))
	
	train_and_evaluate('MLP2', clf = MLPClassifier(hidden_layer_sizes=(2,)))
	train_and_evaluate('MLP10', clf = MLPClassifier(hidden_layer_sizes=(10,)))
	train_and_evaluate('MLP40', clf = MLPClassifier(hidden_layer_sizes=(40,)))
	
	"""
	# Very slow to train...
	C = 0.05
	gamma = 0.05
	#print("Training classifier with SVM-RB with C=%s gamma=%s" % (C, gamma))
	train_and_evaluate('SVM', clf = SVC(C=C, kernel='rbf', gamma=gamma, probability=True))
	
	C = 0.5
	gamma = 0.05
	#print("Training classifier with SVM-RB with C=%s gamma=%s" % (C, gamma))
	train_and_evaluate('SVM', clf = SVC(C=C, kernel='rbf', gamma=gamma, probability=True))

	C = 1
	gamma = 0.05
	#print("Training classifier with SVM-RB with C=%s gamma=%s" % (C, gamma))
	train_and_evaluate('SVM', clf = SVC(C=C, kernel='rbf', gamma=gamma, probability=True))

	C = 5
	gamma = 0.05
	#print("Training classifier with SVM-RB with C=%s gamma=%s" % (C, gamma))
	train_and_evaluate('SVM', clf = SVC(C=C, kernel='rbf', gamma=gamma, probability=True))
	"""
	
