=============================
Sound Commander
=============================

Open source speech recognition is crap, for two reasons.

*Reason 1*: Because it tries to do

	soundwaves -> phonemes -> textual representation -> meaning

Why not solve a simpler problem?

I do not need to have my computer "translate" sounds into text, or "understand" a meaning.

I just want to tell my computer a command and it does something. So I only need 

	soundwaves -> label

*Reason 2*: It is hard to enter for programmers because training sets are limited 
and/or hard to obtain. Machine learning algorithms need a vast amount of data 
to be successful and freely available datasets just aren't there.

-----------
Solution
-----------

In this repo I want to experiment with some machine learning algorithms.
There are three parts

*Part I*: Generating data sets

A very large labeled test training set is generated (generate.sh) by a text-to-speech program (espeak). We are working with phonemes.
I selected all english single-syllable word as labels, obtained their phonemes (thanks to BEEP), and generate audio files varying

* speaker
* stress
* pitch
* speed

For each word there are 1500-2000 samples. There are 5153 words in total.

I also pass the audio through OPUS (formerly speex) with a low bitrate. Because this compression is optimized for speech, hopefully it filters out non-speech. Also, it saves disk space.

*Part II*: Detecting voice (TODO)

Training set: I should generate a stream of audio by stitching together random word audio samples (quiet times cropped) and random lengths of silence (with varying noise properties). A labelling stream should be generated at the same time, identifying silent times. 
Short-time Fourier Transforms (STFT) should be a good input format.

Methods: 

* Any machine learning method can be trained. The method should react as soon as possible however (Recurrent neural network come to mind). 
* The alternative is voice activity detection algorithms. For example implemented in Speex, WebRTC, https://github.com/voixen/voixen-vad. 

*Part III*: Detecting words (TODO)

Training set: Generated in Part I, but should be STFT.

*Part IV*: Putting it together (TODO)

The best method of Part II should detect voice; the audio segment can then be isolated by cropping before and after. The best method of Part III can then be applied to this audio piece, identifying the spoken word. Finally, the command associated with that label can be executed (e.g. just saying the word back).






