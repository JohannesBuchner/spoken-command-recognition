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
There are three parts

I: Generating data sets
----------------------------------

A very large labeled test training set is generated (generate.sh) by a text-to-speech program (espeak). We are working with phonemes.
I selected all english single-syllable word as labels, obtained their phonemes (thanks to BEEP), and generate audio files varying

* speaker
* stress
* pitch
* speed

For each word there are 1500-2000 samples. There are 5153 words in total, 558 in the "verbs" subset.

I also pass the audio through OPUS (formerly speex) with a low bitrate. Because this compression is optimized for speech, hopefully it filters out non-speech. Also, it saves disk space.

II: Detecting voice (TODO)
-----------------------------

Training set: I should generate a stream of audio by stitching together random word audio samples (quiet times cropped) and random lengths of silence (with varying noise properties). A labelling stream should be generated at the same time, identifying silent times. 
Short-time Fourier Transforms (STFT) should be a good input format.

Methods: 

* Any supervised classification machine learning method can be trained. The method should react as soon as possible however (Recurrent neural networks come to mind). 
* The alternative is voice activity detection algorithms. For example implemented in Speex, WebRTC, https://github.com/voixen/voixen-vad. 

III: Detecting words (TODO)
----------------------------------------

Training set: Generated in Part I, but should be STFT.

Methods: 

* Any supervised classification machine learning method can be trained. Convolutional neural networks could be useful.


IV: Putting it together (TODO)
------------------------------------

The best method of Part II should detect voice; the audio segment can then be isolated by cropping before and after. The best method of Part III can then be applied to this audio piece, identifying the spoken word. Finally, the command associated with that label can be executed (e.g. just saying the word back).



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

