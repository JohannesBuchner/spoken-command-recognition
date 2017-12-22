=============================
Sound Commander
=============================

A large database of free audio samples (10M words), a test bed for voice activity detection algorithms and for recognition of syllables (single-word commands).

-----------
Background
-----------

<begin rant>

Open source speech recognition is crap, for two reasons.

**Reason 1**: Because it tries to do everything::

	soundwaves -> phonemes -> textual representation -> meaning

Why not solve a simpler problem?

I do not need to have my computer "translate" sounds into text, or "understand" a meaning.
Why should we insist on having a human-human interaction with a computer? 
We use mice and keyboards, they are not "native language" for human-human interaction. 
Why not also for speech interaction with computers, define and learn a format 
that is easy for a computer?

I just want to tell my computer a command and it does something. So I only need::

	soundwaves -> label

**Reason 2**: It is hard to enter for programmers because training sets are limited 
and/or hard to obtain. Machine learning algorithms need a vast amount of data 
to be successful and freely available datasets just aren't there.

<end rant>

Approach
=========

In this repo I want to experiment with some machine learning algorithms.
There are four parts

I: Generating data sets
----------------------------------

The approach is to build commands off the simplest utterances: single-syllable verbs. 

How to create a large training set? With text-to-speech programs and varying noise sources. Whether well-trained methods can translate to the real world will teach us something about knowledge transfer.
First we create the clean utterances. A very large labeled test training set is generated (generate.sh) by a text-to-speech program (espeak). For simplicity we are working with phonemes instead of words.
I selected all english single-syllable word as labels, obtained their phonemes (thanks to BEEP), and generate audio files varying

* speaker
* stress
* pitch
* speed
* (and further down, noise level and noise type)

For each word there are 1500-2000 samples. There are 5153 words in total, 558 in the "verbs" subset.

I pass the audio through OPUS (formerly speex) with a low constant bitrate. Because this compression (opus was formerly speex) is optimized for speech, hopefully it filters out non-speech to some degree. More importantly, it saves disk space.

II: Detecting voice activity
-------------------------------------

Training set: We generate a very long stream of audio by stitching together random word audio samples (quiet times cropped) and random lengths of silence (with varying noise properties). A labelling stream should be generated at the same time, identifying silent times. 

Build with::

	python trainactivitydetect_prepare.py  verbs.dog db.dog db.dog.hdf5
	python trainactivitydetect_prepare.py  verbs db db.verbs.hdf5

Short-time Fourier Transforms (STFT) should be a good input format for algorithms.

Methods: 

* Any supervised classification machine learning method can be trained. The method should react as soon as possible however (Recurrent neural networks come to mind). 
* The alternative is voice activity detection algorithms. For example implemented in Speex, WebRTC, https://github.com/voixen/voixen-vad. 

Testing algorithms with::

	$ python trainactivitydetect_cv.py out.voiceactivity?-???.hdf5
	...
	memory from the last 280 frames
	loading... out.voiceactivity0-0.1.hdf5
	   (142763, 65) {u'noisesource': 'airport'}
	  attaching memory ...
	    selected 200 200
	  attaching memory done.
	loading... out.voiceactivity0-0.3.hdf5
	   (142763, 65) {u'noisesource': 'airport'}
	  attaching memory ...
	    selected 200 200
	  attaching memory done.
	loading... out.voiceactivity0-0.5.hdf5
	   (142763, 65) {u'noisesource': 'airport'}
	  attaching memory ...
	    selected 200 200
	  attaching memory done.
	loading... out.voiceactivity0-1.0.hdf5
	   (142763, 65) {u'noisesource': 'airport'}
	  attaching memory ...
	    selected 200 200
	  attaching memory done.
	loading... out.voiceactivity1-0.1.hdf5
	   (165485, 65) {u'noisesource': 'babble'}
	  attaching memory ...
	    selected 200 200
	  attaching memory done.
	loading... out.voiceactivity1-0.3.hdf5
	   (165485, 65) {u'noisesource': 'babble'}
	  attaching memory ...
	    selected 200 200
	  attaching memory done.
	loading... out.voiceactivity1-0.5.hdf5
	   (165485, 65) {u'noisesource': 'babble'}
	  attaching memory ...
	    selected 200 200
	  attaching memory done.
	loading... out.voiceactivity1-1.0.hdf5
	   (165485, 65) {u'noisesource': 'babble'}
	  attaching memory ...
	    selected 200 200
	  attaching memory done.
	loading... out.voiceactivity2-0.1.hdf5
	   (152073, 65) {u'noisesource': 'brown'}
	  attaching memory ...
	    selected 200 200
	  attaching memory done.
	loading... out.voiceactivity2-0.3.hdf5
	   (152073, 65) {u'noisesource': 'brown'}
	  attaching memory ...
	    selected 200 200
	  attaching memory done.
	loading... out.voiceactivity2-0.5.hdf5
	   (152073, 65) {u'noisesource': 'brown'}
	  attaching memory ...
	    selected 200 200
	  attaching memory done.
	loading... out.voiceactivity2-1.0.hdf5
	   (152073, 65) {u'noisesource': 'brown'}
	  attaching memory ...
	    selected 200 200
	  attaching memory done.
	loading... out.voiceactivity3-0.1.hdf5
	   (153851, 65) {u'noisesource': 'car'}
	  attaching memory ...
	    selected 200 200
	  attaching memory done.
	loading... out.voiceactivity3-0.3.hdf5
	   (153851, 65) {u'noisesource': 'car'}
	  attaching memory ...
	    selected 200 200
	  attaching memory done.
	loading... out.voiceactivity3-0.5.hdf5
	   (153851, 65) {u'noisesource': 'car'}
	  attaching memory ...
	    selected 200 200
	  attaching memory done.
	loading... out.voiceactivity3-1.0.hdf5
	   (153851, 65) {u'noisesource': 'car'}
	  attaching memory ...
	    selected 200 200
	  attaching memory done.
	loading... out.voiceactivity4-0.1.hdf5
	   (153975, 65) {u'noisesource': 'exhibition'}
	  attaching memory ...
	    selected 200 200
	  attaching memory done.
	loading... out.voiceactivity4-0.3.hdf5
	   (153975, 65) {u'noisesource': 'exhibition'}
	  attaching memory ...
	    selected 200 200
	  attaching memory done.
	loading... out.voiceactivity4-0.5.hdf5
	   (153975, 65) {u'noisesource': 'exhibition'}
	  attaching memory ...
	    selected 200 200
	  attaching memory done.
	loading... out.voiceactivity4-1.0.hdf5
	   (153975, 65) {u'noisesource': 'exhibition'}
	  attaching memory ...
	    selected 200 200
	  attaching memory done.
	loading... out.voiceactivity5-0.1.hdf5
	   (146957, 65) {u'noisesource': 'ocean'}
	  attaching memory ...
	    selected 200 200
	  attaching memory done.
	loading... out.voiceactivity5-0.3.hdf5
	   (146957, 65) {u'noisesource': 'ocean'}
	  attaching memory ...
	    selected 200 200
	  attaching memory done.
	loading... out.voiceactivity5-0.5.hdf5
	   (146957, 65) {u'noisesource': 'ocean'}
	  attaching memory ...
	    selected 200 200
	  attaching memory done.
	loading... out.voiceactivity5-1.0.hdf5
	   (146957, 65) {u'noisesource': 'ocean'}
	  attaching memory ...
	    selected 200 200
	  attaching memory done.
	loading... out.voiceactivity6-0.1.hdf5
	   (141459, 65) {u'noisesource': 'pink'}
	  attaching memory ...
	    selected 200 200
	  attaching memory done.
	loading... out.voiceactivity6-0.3.hdf5
	   (141459, 65) {u'noisesource': 'pink'}
	  attaching memory ...
	    selected 200 200
	  attaching memory done.
	loading... out.voiceactivity6-0.5.hdf5
	   (141459, 65) {u'noisesource': 'pink'}
	  attaching memory ...
	    selected 200 200
	  attaching memory done.
	loading... out.voiceactivity6-1.0.hdf5
	   (141459, 65) {u'noisesource': 'pink'}
	  attaching memory ...
	    selected 200 200
	  attaching memory done.
	loading... out.voiceactivity7-0.1.hdf5
	   (138663, 65) {u'noisesource': 'restaurant'}
	  attaching memory ...
	    selected 200 200
	  attaching memory done.
	loading... out.voiceactivity7-0.3.hdf5
	   (138663, 65) {u'noisesource': 'restaurant'}
	  attaching memory ...
	    selected 200 200
	  attaching memory done.
	loading... out.voiceactivity7-0.5.hdf5
	   (138663, 65) {u'noisesource': 'restaurant'}
	  attaching memory ...
	    selected 200 200
	  attaching memory done.
	loading... out.voiceactivity7-1.0.hdf5
	   (138663, 65) {u'noisesource': 'restaurant'}
	  attaching memory ...
	    selected 200 200
	  attaching memory done.
	loading... out.voiceactivity8-0.1.hdf5
	   (139677, 65) {u'noisesource': 'street'}
	  attaching memory ...
	    selected 200 200
	  attaching memory done.
	loading... out.voiceactivity8-0.3.hdf5
	   (139677, 65) {u'noisesource': 'street'}
	  attaching memory ...
	    selected 200 200
	  attaching memory done.
	loading... out.voiceactivity8-0.5.hdf5
	   (139677, 65) {u'noisesource': 'street'}
	  attaching memory ...
	    selected 200 200
	  attaching memory done.
	loading... out.voiceactivity8-1.0.hdf5
	   (139677, 65) {u'noisesource': 'street'}
	  attaching memory ...
	    selected 200 200
	  attaching memory done.
	loading... out.voiceactivity9-0.1.hdf5
	   (142587, 65) {u'noisesource': 'subway'}
	  attaching memory ...
	    selected 200 200
	  attaching memory done.
	loading... out.voiceactivity9-0.3.hdf5
	   (142587, 65) {u'noisesource': 'subway'}
	  attaching memory ...
	    selected 200 200
	  attaching memory done.
	loading... out.voiceactivity9-0.5.hdf5
	   (142587, 65) {u'noisesource': 'subway'}
	  attaching memory ...
	    selected 200 200
	  attaching memory done.
	loading... out.voiceactivity9-1.0.hdf5
	   (142587, 65) {u'noisesource': 'subway'}
	  attaching memory ...
	    selected 200 200
	  attaching memory done.
	data: (15959, 2624) (15959,)

	with preprocessor log:

	0.87 RandomForest1 (training speed: 15.6s)
	confusion matrix: (eval speed: 0.94s)
	[[2384  293]
	 [ 169 2474]]
	ROC curve plot...
	5% FPR: at threshold 2.0 with efficiency 0.0
	1% FPR: at threshold 2.0 with efficiency 0.0

	0.94 RandomForest4 (training speed: 45.4s)
	confusion matrix: (eval speed: 0.98s)
	[[2564  113]
	 [ 159 2484]]
	ROC curve plot...
	5% FPR: at threshold 0.75 with efficiency 93.98%
	1% FPR: at threshold 1.0 with efficiency 80.85%

	0.94 RandomForest10 (training speed: 106.0s)
	confusion matrix: (eval speed: 1.38s)
	[[2583   94]
	 [ 106 2537]]
	ROC curve plot...
	5% FPR: at threshold 0.6 with efficiency 95.99%
	1% FPR: at threshold 0.8 with efficiency 89.94%

	0.94 RandomForest40 (training speed: 326.7s)
	confusion matrix: (eval speed: 2.95s)
	[[2613   64]
	 [  66 2577]]
	ROC curve plot...
	5% FPR: at threshold 0.45 with efficiency 98.45%
	1% FPR: at threshold 0.625 with efficiency 95.95%


	0.94 AdaBoost (training speed: 1808.8s)
	confusion matrix: (eval speed: 8.24s)
	[[2570  107]
	 [ 102 2541]]
	ROC curve plot...
	5% FPR: at threshold 0.496241695927 with efficiency 96.78 %
	1% FPR: at threshold 0.51368077231 with efficiency 93.15 %

	0.93 GradientBoosting (training speed: 3338.6s)
	confusion matrix: (eval speed: 1.06s)
	[[2522  155]
	 [  76 2567]]
	ROC curve plot...
	5% FPR: at threshold 0.536277698584 with efficiency 96.78 %
	1% FPR: at threshold 0.764611974148 with efficiency 92.47 %

	0.78 MLP2 (training speed: 238.6s)
	confusion matrix: (eval speed: 1.97s)
	[[2498  179]
	 [ 196 2447]]
	ROC curve plot...
	5% FPR: at threshold 0.52487468149 with efficiency 91.87 %
	1% FPR: at threshold 0.634618783147 with efficiency 86.76 %

	0.91 MLP10 (training speed: 306.5s)
	confusion matrix: (eval speed: 3.29s)
	[[2671    6]
	 [ 545 2098]]
	ROC curve plot...
	5% FPR: at threshold 0.269476241912 with efficiency 90.39 %
	1% FPR: at threshold 0.361535546018 with efficiency 85.74 %

	0.91 MLP40 (training speed: 724.1s)
	confusion matrix: (eval speed: 7.18s)
	[[2538  139]
	 [ 198 2445]]
	ROC curve plot...
	5% FPR: at threshold 0.506125804557 with efficiency 92.40 %
	1% FPR: at threshold 0.641257552208 with efficiency 87.44 %

	with preprocessor logscale:

	0.69 MLP2 (training speed: 256.3s)
	confusion matrix: (eval speed: 2.26s)
	[[ 225 2452]
	 [   0 2643]]
	ROC curve plot...
	5% FPR: at threshold 1.52995962594 with efficiency 0.0 %
	1% FPR: at threshold 1.52995962594 with efficiency 0.0 %

	0.92 MLP10 (training speed: 476.6s)
	confusion matrix: (eval speed: 3.61s)
	[[2624   53]
	 [ 315 2328]]
	ROC curve plot...
	5% FPR: at threshold 0.432683623862 with efficiency 91.49 %
	1% FPR: at threshold 0.546176133175 with efficiency 86.11 %

	0.94 MLP40 (training speed: 644.6s)
	confusion matrix: (eval speed: 6.99s)
	[[2653   24]
	 [ 370 2273]]
	ROC curve plot...
	5% FPR: at threshold 0.376385485975 with efficiency 91.56 %
	1% FPR: at threshold 0.496414547621 with efficiency 86.08 %


In conclusion, Random Forest (40 classifiers) does a good job. 
For low false-positive rates, need to set probability score threshold to >0.625, 

MLP also has a comparable quality.

III: Detecting words (TODO)
----------------------------------------

Training set: Generated in Part I, but should be STFT.

Methods: 

* Any supervised classification machine learning method can be trained. Convolutional neural networks could be useful.


IV: Putting it together (TODO)
------------------------------------

The best method of Part II should detect voice; the audio segment can then be isolated by cropping before and after. The best method of Part III can then be applied to this audio piece, identifying the spoken word. 
Finally, the command associated with that label can be executed. 

For example, the computer could just be saying the identified word back, or run a program, shut down, change the speaker volume, etc.



How to use
=============

Generating the dataset
-----------------------

Use generateverbs.sh to generate the sound files for the "verbs" subset::

	$ bash generateverbs.sh

This will create files like db.verbs/<word>/<variant>
where for example word="aim" and variant="132" is one pronounciation of that word.
These are opus sound files. 
This takes a while (10 minutes per word). You need: opusenc, espeak, mbrola and sox.
The size of the database will be approximately 10MB per word.

To check a example pronounciation, play a random word like this::

	$ w=$(ls db|sort -R|head -n1); p=$(ls db/$w|sort -R|head -n1); play db/$w/$p

Some words are pretty hard to understand, some are quite easy. Don't think of these classes as words. 
Think of them as groups of utterances that humans and computers can both agree to assign a single meaning to.

To find out for a given variant (e.g. 753), how it was produced (which speaker, pitch, speed), use::

	$ bash versionnames.sh |grep -w 753
	753 english_wmids 140 70


Importing the dataset into Python
----------------------------------

To create a numpy array containing the spectrograms, run::

	$ bash generatenoise.sh # to generate or download the noise files that will be added
        $ python traincommanddetect.py

The output file is db.verbs.npz, a numpy compressed array, with the keys "audiodata" and "labels".
The data shape is (nsamples=many, nframes=30, nspectralbins=513).


Training the method
----------------------------------

Your turn. Go wild.

