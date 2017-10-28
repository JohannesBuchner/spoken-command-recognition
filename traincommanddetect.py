"""

build training set for word recognition methods

"""

import random
import sys, os
import numpy
import scipy
import scipy.io.wavfile as wav
import stft
import subprocess
import itertools
import h5py
import matplotlib.pyplot as plt
from PIL import Image

numpy.random.seed(1)

verbs = [row.strip().split()[0] for row in open('verbs')]

framelength = 128
def wav_to_spectrogram(filename):
	# 8000 Hz sampling
	# framelength of 128 gives
	# 0-4000 Hz range in spectral bins
	fs, audio = wav.read(filename)
	specgram = stft.spectrogram(audio, framelength=framelength)
	specgramabs = numpy.abs(specgram).transpose()
	#print specgramabs.shape
	return specgramabs
	

# generate some noise files and read frames from them on-demand
noisenames = 'airport babble brown car exhibition ocean pink restaurant street subway train white'.split()
noises = {}
for noise in noisenames:
	print 'loading noise "%s" ...' % noise
	# load file and make SFTF frames
	specgramabs = wav_to_spectrogram('db.noise/%s.wav' % noise)
	# set the frames on auto-repeat with itertools
	noises[noise] = itertools.cycle(specgramabs / specgramabs.max())

print '%-10s\tvariant\tvolume\tnoisetype\tnoisevolume' % ('word')
maxlength = 30
imgshape = (24, 24)
labels = []
data = []
for i, word in enumerate(verbs):
	data_this_word = []
	for entry in os.listdir(os.path.join('db.verbs', word)):
		wordfile = os.path.join('db.verbs', word, entry)
		# read in word file, make SFTF
		p = subprocess.Popen(['sox', wordfile, '-t', 'wav', '-r', '8000', '-c', '1', 'tmp.wav'])
		#p = subprocess.Popen(['sox', wordfile, '-t', 'wav', 'tmp.wav'])
		#p = subprocess.Popen(['sox', wordfile, '-t', 'wav', '-r', '8000', '-c', '1', 'tmp.wav', 'silence', '1', '0.1', '0.1%', 'reverse', 'silence', '1', '0.1', '0.1%', 'reverse'])
		p.wait()
		#print wordfile
		#fs, audio = wav.read('tmp.wav')
		specgramabs = wav_to_spectrogram('tmp.wav')
		#if len(audio) == 0: continue
		#specgram = stft.spectrogram(audio, framelength=framelength)
		#specgramabs = numpy.abs(specgram).transpose()
		#print specgramabs.shape
		voicepart = specgramabs[:,8:]
		voiceactive = voicepart.sum(axis=1) > voicepart.sum(axis=1).max() * 1e-3
		end = numpy.where(voiceactive)[0].max()
		#print voiceactive, end
		#plt.figure()
		#plt.imshow(numpy.log10(specgramabs))
		#plt.show()
		voicepart = specgramabs[:end,:]
		#specgramabs /= specgramabs.max()
		#print specgramabs.shape
		nframes, nspec = voicepart.shape
		#print nframes, nspec, voicepart.max()
		
		#assert nframes < maxlength, (nframes, maxlength)
		#full = numpy.zeros((maxlength, nspec))
		#full[:nframes,:] = specgramabs
		## The following could be repeated several times to increase the sample size

		## modify volume by multiplying SFTF amplitudes
		#sourcevolume = random.normalvariate(1,0.2)

		#combination = full * sourcevolume
		# choose a random noise source
		noisesource = random.choice(noises.keys())
		noisevolume = random.uniform(0, 0.03)
		# stack onto SFTF frames the noise frames, scaled
		#for i in range(maxlength):
		#	combination[i,:] += noisevolume * next(noises[noisesource])

		for i in range(nframes):
			voicepart[i,:] += noisevolume * next(noises[noisesource])
		
		# 0Hz, 60Hz, 120Hz, ... - 4000Hz are the frequencies we stored
		# we store 500Hz upwards, which is audible by humans and related to speech
		img = voicepart[:,8:]
		# reshape to common shape (e.g. 24x24 pixels, 256 colors)
		# now normalise to 1 and take logarithms
		img = (numpy.log(voicepart / voicepart.max() * 0.99 + 1e-10) + 255).astype('uint8')
		img = scipy.misc.imresize(img, size=imgshape, mode='F')
		print '%-10s\t%s\t%-20s\t%.2f' % (word, entry, noisesource, noisevolume)
		data.append(img)
		data_this_word.append(img)
		labels.append(i)
	
	plt.figure(figsize=(20,20))
	plt.suptitle(word)
	plotentries = random.sample(data_this_word, 25)
	for i, img in enumerate(plotentries):
		plt.subplot(int(numpy.ceil(len(plotentries) / 5.)), 5, i)
		plt.imshow(img, cmap='RdBu')
	plt.savefig('db.verbs.%s.png' % word, bbox_inches='tight')
	plt.close()
	#break
with h5py.File('db.verbs.hdf5', 'w') as f:
	f.create_dataset('data', data=data, shuffle=True, compression='gzip')
	f.create_dataset('labels', data=labels, shuffle=True, compression='gzip')

