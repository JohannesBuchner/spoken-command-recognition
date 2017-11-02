"""

Use Keras and a simple Long short-memory neural network to detect voice activity


Notes: RandomForest and MLP10 do very well (95% correctness) with log preprocessor
Worse if only current sample is considered (90%)
Worse if only current sample and 3s median is considered (91%)



"""

import sys, os
import numpy
import h5py
import math
import scipy
from time import time
from sklearn import preprocessing
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.decomposition import PCA
from sklearn.svm import SVC
import matplotlib.pyplot as plt

numpy.random.seed(1)

print 'Command detection:'
print '  Loading command recognition training set...'
infile = sys.argv[1]

f = h5py.File(infile, 'r')
labels = f['labels'].value
nsamples, imgh, imgw = f['data'].shape
imgshape = (imgh, imgw)
#for i in numpy.random.randint(len(labels), size=5):
#	print f['data'][i,:,:]
data = f['data'].value.reshape((len(labels), -1))


X_train = data
print("  Dimensionality reduction with PCA...")
n_components = 16
t0 = time()
pca = PCA(n_components=n_components, svd_solver='randomized',
          whiten=True).fit(X_train)
print("  Dimensionality reduction done (%0.3fs)" % (time() - t0))

X_train_pca = pca.transform(X_train)

print("  Training command classifier with SVM-RB ...")
C = 0.1
gamma = 0.05
t0 = time()
clf = SVC(C=C, kernel='rbf', gamma=gamma)
clf = clf.fit(X_train_pca, labels)
#clf = RandomForestClassifier(n_estimators=100)
#clf = clf.fit(X_train_pca, labels)
print("  Training command classifier done (%0.3fs)" % (time() - t0))

def detect_meaning(voicepart):
	img = voicepart[:,8:]
	# reshape to common shape (e.g. 24x24 pixels, 256 colors)
	# now normalise to 1 and take logarithms
	img = (numpy.log(voicepart / voicepart.max() * 0.99 + 1e-10) + 255).astype('uint8')
	img = scipy.misc.imresize(img, size=imgshape, mode='F')
	label = clf.predict(pca.transform(img.reshape(1,-1)))
	return img, label

print 'Voice activity detection:'
print '  loading voice activity detector training set ...'
with h5py.File('voicedetect-training.hdf5') as f:
	X = f['x7'].value
	Y = f['y'].value
	#print Y.mean(), Y.shape, X.shape

print '  training voice activity detector ...'
t0 = time()
vadclf = RandomForestClassifier(n_estimators=10)
#clf = MLPClassifier(hidden_layer_sizes=(10,))
vadclf = vadclf.fit(X, Y)
print '  training voice activity detector done (%.1fs)' % (time() - t0)

print 'Running:'
filename = sys.argv[2]
print '  loading test data set...', filename
f = h5py.File(filename, 'r')
labels = f['labels'].value
data = f['data'].value
print '  ', data.shape, dict(f.attrs)


look_back_seconds = 2
# files are 150000 samples long and cover ~20minutes, so it is
# approximately 140 samples per second
look_back = int(140 * look_back_seconds)
stride = 7
print 'memory from the last %d frames' % look_back
indices = []
istart = None
ioff = None
j = 1
for i in range(look_back, len(data)):
	dataset = data[i-look_back:i+1,:64][::-1][::stride][::-1]
	dataset2 = (numpy.log(dataset + 1e-3) - numpy.log(1e-3)) / 30.
	y = vadclf.predict(dataset2.flatten().reshape((1,-1)))
	#print i, labels[i], y, numpy.log10(dataset[-1,::8]).astype(int)
	if y > 0:
		if istart is None:
			# starting segment
			istart = i
		# continuing segment
		ioff = i
	else:
		if ioff and istart and i > ioff + 3 and ioff > istart + 10:
			# emit segment
			img, label = detect_meaning(data[istart:ioff,:])
			print '  found some activity:', istart, ioff, label
			plt.title(label)
			plt.imshow(img, cmap='RdBu')
			plt.savefig('example.%d.png' % j, bbox_inches='tight')
			plt.close()
			ioff = None
			istart = None
			j += 1
		
	#print labels[i], y



