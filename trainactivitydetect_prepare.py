"""

building sentences (long audio files)
for training voice activity detection methods.

"""

import random
import sys, os
import numpy
import subprocess
import stft
import joblib
import scipy, scipy.stats
import scipy.io.wavfile as wav
import h5py

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

def generate_silence_length():
	while True:
		length = random.expovariate(1./10)
		if length < 100 and length > 0.5: 
			return length

def run(cmd):
	# shell=True is not particularly secure, if filenames are crafted
	subprocess.check_call(cmd, shell=True)

wordfile = sys.argv[1]
worddir = sys.argv[2]
outfile = sys.argv[3]

words = [row.strip().split()[0] for row in open(wordfile)]

def generate_word():
	wordi = random.randint(0, len(words)-1)
	word = words[wordi]
	entries = os.listdir(os.path.join(worddir, word))
	entry = random.choice(entries)
	volume = random.normalvariate(1,0.2)
	return (wordi, word, os.path.join(worddir, word, entry), volume)


# build a sentence
noisenames = 'airport babble brown car exhibition ocean pink restaurant street subway train white'.split()
#noisesource = random.choice(noisenames)
random.seed(1)

for j, noisesource in enumerate(noisenames):
	#nwords = numpy.random.poisson(numpy.random.poisson(10)+1)
	nwords = 100
	sentence = []
	sentence.append(('silence', generate_silence_length()))
	for i in range(nwords):
		sentence.append(('word', generate_word()))
		sentence.append(('silence', generate_silence_length()))

	# build audio file with sox
	commands = []
	files_clean = []
	data = []
	labels = []
	labels_words_used = []
	files_reference = []

	for i, part in enumerate(sentence):
		filename = 'part%d.wav' % i
		reffilename = 'part%dref.wav' % i
		if part[0] == 'silence':
			_silence, length = part
			reffilename = filename
			print '  %.2fs silence' % length
			commands.append('sox -n -r 48000 -c 1 %s trim 0.0 %.2f' % (filename, length))
		elif part[0] == 'word':
			_word, (wordi, word, wordfile, volume) = part
			labels_words_used.append(wordi)
			print '  %s  vol=%.2f' % (word, volume)
			#commands.append('sox -v %.2f %s %s silence 1 0.1 0.1%% reverse silence 1 0.1 0.1%% reverse' % (volume, wordfile, filename))
			#commands.append('sox %s -p synth whitenoise vol 1 | sox -m %s - %s' % (filename, filename, reffilename))
			commands.append('sox -v %.2f %s %s silence 1 0.1 0.1%% reverse silence 1 0.1 0.1%% reverse; sox %s -p synth whitenoise vol 1 | sox -m %s - %s' % (volume, wordfile, filename, filename, filename, reffilename))
		files_clean.append(filename)
		files_reference.append(reffilename)
	
	print 'building audio parts ...'
	joblib.Parallel(n_jobs=-1)(joblib.delayed(run)(cmd) for cmd in commands)
	#[run(cmd) for cmd in commands]
	
	# this file contains voice activity labelling. silence if inactive, loud white noise if active
	sentence_reference = 'sentence%d_reference.wav' % j
	# this file contains the spoken words with silence in between, without noise
	sentence_clean = 'sentence%d_clean.wav' % j
	# same, but with noise added
	sentence_noisy = 'sentence%d_noisy.wav' % j
	
	p1 = subprocess.Popen(['sox'] + files_reference + ['-r', '8000', '-c', '1', sentence_reference])
	subprocess.check_call(['sox'] + files_clean + ['-r', '8000', '-c', '1', sentence_clean])
	p1.wait()
	[os.unlink(f) for f in set(files_clean + files_reference)]

	# add noise to the clean file
	if noisesource in ['restaurant', 'street', 'white', 'street', 'babble']:
		noisevolumeceil = 0.2
	else:
		noisevolumeceil = 0.5
		#noisevolume = random.normalvariate(1, 0.2)
	
	for noisevolumerel in [0.1, 0.3, 0.5, 1]:
		noisevolume = noisevolumerel * noisevolumeceil

		print 'adding noise ...', noisesource, noisevolume
		outlength = subprocess.check_output(['sox', '--i', '-d', sentence_clean])
		subprocess.check_call(['sox', '-m', sentence_clean, '|sox db.noise/%s.wav -p vol %.2f repeat 100 trim 0 %s' % (noisesource, noisevolume, outlength), sentence_noisy])
	
		print 'reading reference data...'
		data_reference = wav_to_spectrogram(sentence_clean)
		activity = data_reference.sum(axis=1)
		# mark the faint end
		lo, = scipy.stats.mstats.mquantiles(activity[activity>0], 0.05)
		labels = (activity > lo)*1
		#import matplotlib.pyplot as plt
		#plt.hist(numpy.log10(activity + 1e-2), bins=100)
		#plt.show()
		data_reference = wav_to_spectrogram(sentence_reference)
		activity = data_reference.sum(axis=1)
		labels += (activity > 0)*1
		
		print 'reading noisy data...'
		data_noisy = wav_to_spectrogram(sentence_noisy)
		print labels.shape, data_reference.shape, data_noisy.shape
	
		f = h5py.File('%s%d-%.1f.hdf5' % (outfile, j, noisevolumerel), 'w')
		f.create_dataset('data', data=data_noisy, shuffle=True, compression='gzip')
		f.create_dataset('labels', data=labels, shuffle=True, compression='gzip')
		f.create_dataset('labels_words', data=labels_words_used, shuffle=True, compression='gzip')
		f.attrs['noise-source'] = noisesource
		f.attrs['noise-volume-normalisation'] = noisevolumeceil
		f.attrs['noise-volume-relative'] = noisevolumerel
	






