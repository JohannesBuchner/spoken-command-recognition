"""

build training set for word recognition methods

"""

import random
import sys, os
import numpy
import scipy.io.wavfile as wav
import stft
import subprocess
import itertools

numpy.random.seed(1)

verbs = [row.strip().split()[0] for row in open('verbs')]

# generate some noise files and read frames from them on-demand
noisenames = 'airport babble brown car exhibition ocean pink restaurant street subway train white'.split()
noises = {}
for noise in noisenames:
        print 'loading noise "%s" ...' % noise
	# load file and make SFTF frames
        fs, audio = wav.read('db.noise/%s.wav' % noise)
        specgram = stft.spectrogram(audio)
	# set the frames on auto-repeat with itertools
        specgramabs = numpy.abs(specgram).transpose()
	noises[noise] = itertools.cycle(specgramabs / specgramabs.max())

print '%-10s\tvariant\tvolume\tnoisetype\tnoisevolume' % ('word')
data = []
labels = []
maxlength = 30
for i, word in enumerate(verbs):
	for entry in os.listdir(os.path.join('db.verbs', word)):
		wordfile = os.path.join('db.verbs', word, entry)
		# read in word file, make SFTF
                p = subprocess.Popen(['sox', wordfile, '-t', 'wav', '-r', '8000', '-c', '1', 'tmp.wav'])
                p.wait()
                fs, audio = wav.read('tmp.wav')
                specgram = stft.spectrogram(audio)
                specgramabs = numpy.abs(specgram).transpose()
                specgramabs /= specgramabs.max()
                #print specgramabs.shape
		nframes, nspec = specgramabs.shape

                assert nframes < maxlength, (nframes, maxlength)
		full = numpy.zeros((maxlength, nspec))
                full[:nframes,:] = specgramabs
                # The following could be repeated several times to increase the sample size
                   
                # modify volume by multiplying SFTF amplitudes
		sourcevolume = random.normalvariate(1,0.2)
	        
                combination = full * sourcevolume
                # choose a random noise source
		noisesource = random.choice(noises.keys())
		noisevolume = random.uniform(0, 0.03)
		# stack onto SFTF frames the noise frames, scaled
                for i in range(maxlength):
                    combination[i,:] += noisevolume * next(noises[noisesource])
                print '%-10s\t%s\t%.2f\t%-20s\t%.2f' % (word, entry, sourcevolume, noisesource, noisevolume)
                data.append(combination)
		labels.append(i)
	break	

numpy.savez('db.verbs.npz', audiodata=data, labels=labels)

