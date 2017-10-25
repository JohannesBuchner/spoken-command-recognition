"""

build training set for word recognition methods

"""

import random
import sys, os
import numpy

# TODO: read in sentence_noisy.wav and make SFTF from frames
# TODO: read in sentence_reference.wav and label silent + non-silent frames
# TODO: pass the two to a machine-learning method

words = [row.strip().split()[0] for row in open('words')]

# generate some noise files and read frames from them on-demand
noises = {}
for noise in ['whitenoise', 'pinknoise', 'brownnoise']:
	# TODO: generate very long sox file and read frames here
	noises[noise] = []

data = []
labels = []

for i, word in enumerate(words):
	for entry in os.listdir(os.path.join('db', word)):
		wordfile = os.path.join('db', word, entry), volume)
		# TODO: read in word file, make SFTF
		# TODO: modify volume by multiplying SFTF amplitudes
		volume = random.normalvariate(1,0.2)
		# TODO: choose a random noise source
		noisesource = random.choice(['whitenoise', 'pinknoise', 'brownnoise'])
		v = random.uniform(0, 0.03)
		# TODO: stack onto SFTF frames the noise frames, scaled
		
		# TODO: add frames to dataset
		data.append(None)
		labels.append(i)
		



