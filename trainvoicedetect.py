"""

building sentences (long audio files)
for training voice activity detection methods.

"""

import random
import sys, os
import numpy

words = [row.strip().split()[0] for row in open('words') if row.startswith('a')]

# build a sentence
nwords = numpy.random.poisson(numpy.random.poisson(10)+1)

def generate_silence_length():
	while True:
		length = 1/(random.paretovariate(0.5))
		if length < 100: 
			return length

def generate_word():
	word = random.choice(words)
	entries = os.listdir(os.path.join('db', word))
	entry = random.choice(entries)
	volume = random.normalvariate(1,0.2)
	return (word, os.path.join('db', word, entry), volume)

sentence = []
sentence.append(('silence', generate_silence_length()))
for i in range(nwords):
	sentence.append(('word', generate_word()))
	sentence.append(('silence', generate_silence_length()))

print sentence
print

# build audio file with sox
commands = []
files_clean = []
files_reference = []
for i, part in enumerate(sentence):
	filename = 'part%d.wav' % i
	reffilename = 'part%dref.wav' % i
	if part[0] == 'silence':
		_silence, length = part
		reffilename = filename
		commands.append('sox -n -r 48000 -c 1 %s trim 0.0 %.2f' % (filename, length))
	elif part[0] == 'word':
		_word, (word, wordfile, volume) = part
		commands.append('sox -v %.2f %s %s' % (volume, wordfile, filename))
		commands.append('sox %s -p synth whitenoise vol 1 | sox -m %s - %s' % (wordfile, wordfile, reffilename))
	
	files_clean.append(filename)
	files_reference.append(reffilename)
print '\n'.join(commands)
print
print 'sox ' + ' '.join(files_reference) + ' sentence_reference.wav'

# add noise to the clean file

noise = random.choice(['whitenoise', 'pinknoise', 'brownnoise'])
v = random.uniform(0, 0.03)
print 'sox ' + ' '.join(files_clean) + ' sentence_clean.wav'
print 'sox sentence_clean.wav -p synth %s vol %s | sox -m sentence_clean.wav - sentence_noisy.wav' % (noise, v)





