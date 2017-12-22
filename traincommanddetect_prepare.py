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
import joblib
import tempfile
import matplotlib.pyplot as plt
from PIL import Image

numpy.random.seed(1)

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

def preprocess_word(wordfile, noiseframes):
	# read in word file, make SFTF
	print 'processing %s' % wordfile
	fd, tmpfile = tempfile.mkstemp('.wav')
	p = subprocess.Popen(['sox', wordfile, '-t', 'wav', '-r', '8000', '-c', '1', tmpfile])
	p.wait()
	specgramabs = wav_to_spectrogram(tmpfile)
	os.unlink(tmpfile)
	voicepart = specgramabs[:,8:]
	voiceactive = voicepart.sum(axis=1) > voicepart.sum(axis=1).max() * 1e-3
	end = numpy.where(voiceactive)[0].max()
	nallframes, nspec = specgramabs.shape
	# pad with empty frames at the beginning and back
	# extend duration by up to 40%
	# cut off up to 10% at the beginning and end
	voice_start = numpy.random.randint(0, int(end * 0.1))
	voice_end = numpy.random.randint(int(end*0.9), nallframes)
	voice_start2 = numpy.random.randint(0, int(end * 0.1))
	nframes_max = max(int(nallframes * 1.4), voice_start2+voice_end-voice_start)
	nframes_min = min(int(nallframes * 1.4), voice_start2+voice_end-voice_start)
	nframes = numpy.random.randint(nframes_min, nframes_max)
	voicepart = numpy.zeros((nframes, nspec))
	#print nallframes, voice_start, voice_end, nframes, voice_start2
	voicepart[voice_start2:voice_start2+voice_end-voice_start,:] = specgramabs[voice_start:voice_end,:]
	# stack onto SFTF frames the noise frames, scaled
	for i in range(nframes):
		voicepart[i,:] += noiseframes[i]
	# 0Hz, 60Hz, 120Hz, ... - 4000Hz are the frequencies we stored
	# we store 500Hz upwards, which is audible by humans and related to speech
	img = voicepart[:,8:]
	# reshape to common shape (e.g. 24x24 pixels, 256 colors)
	# now normalise to 1 and take logarithms
	img = (numpy.log(voicepart / voicepart.max() * 0.99 + 1e-10) + 255).astype('uint8')
	img = scipy.misc.imresize(img, size=imgshape, mode='F')
	return img

def generate_noise_frames(nframes=512):
	noisesource = random.choice(noises.keys())
	noisevolume = random.uniform(0, 0.03)
	for i in range(nframes):
		yield noisevolume * next(noises[noisesource])


print 'scanning directory...'
wordfile = sys.argv[1]
worddir = sys.argv[2]
outfile = sys.argv[3]
all_words = []
labels = []
verbs = [row.strip().split()[0] for row in open(wordfile)]
for i, word in enumerate(verbs):
	word_entries = [os.path.join(worddir, word, entry) for entry in sorted(os.listdir(os.path.join(worddir, word)), key=int)]
	labels += [i] * len(word_entries)
	all_words.append((word, word_entries))
print 'scanning directory done.'

f = h5py.File(outfile, 'w')
f.create_dataset('data', shape=(len(labels), imgshape[0], imgshape[1]))
f.create_dataset('labels', data=labels, shuffle=True, compression='gzip')

j = 0
for i, (word, wordfiles) in list(enumerate(all_words)):
	noise = [list(generate_noise_frames()) for _ in wordfiles]
	
	data_this_word = joblib.Parallel(n_jobs=-1)(joblib.delayed(preprocess_word)(wordfile, noiseframes) for wordfile, noiseframes in zip(wordfiles, noise))
	#data_this_word = [preprocess_word(wordfile, noiseframes) for wordfile, noiseframes in zip(wordfiles, noise)]
	
	for i, img in enumerate(data_this_word):
		f['data'][j,:,:] = img 
		j = j + 1
	
	
	print 'plotting for %s...' % word
	plt.figure(figsize=(20,20))
	plt.suptitle(word)
	plotentries = random.sample(data_this_word, 25)
	for i, img in enumerate(plotentries):
		plt.subplot(int(numpy.ceil(len(plotentries) / 5.)), 5, i)
		plt.imshow(img, cmap='RdBu')
	plt.savefig('db.verbs.%s.png' % word, bbox_inches='tight')
	plt.close()
	print 'plotting done.'
	#break
#with h5py.File('db.verbs.hdf5', 'w') as f:
#	f.create_dataset('data', data=data, shuffle=True, compression='gzip')
#	f.create_dataset('labels', data=labels, shuffle=True, compression='gzip')

