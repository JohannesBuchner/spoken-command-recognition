===================================
Synthetic Speech Commands dataset
===================================

-----------
Background
-----------

* We would like to have good open source speech recognition
* Commerical companies trying to solve a hard problem (map arbitrary, open-ended speech to text and identify meaning). 
* The easier problem should be detecting a predefined word and mapping it to a predefined action.
* Lets tackle the simplest problem first: Classifying single, short words (commands)
* Audio training data is difficult to obtain.


-----------
Approaches
-----------

* The parent project creates synthetic speech datasets using text-to-speech programs. The focus is on single-syllable verbs (commands).
* The Speech Commands dataset asked volunteers to pronounce a small set of words: (yes, no, up, down, left, right, on, off, stop, go, and 0-9).
* This project provides synthetic counterparts to this real world dataset.

---------------
Open questions
---------------

One can use these datasets in various ways. Here are some things I am interested in seeing answered:

1. What is it in an audio sample that makes it "sound similar"?
   Our ears can easily classify both synthetic and real speech, but for algorithms this is still hard.
   Extending the real dataset with the synthetic data yields a larger training sample and more diversity.

2. How well does an algorithm trained on one data set perform on the other? (transfer learning)
   If it works poorly, the algorithm probably has not found the key to audio similarity.
   
3. Can an algorithm trained on synthetic data classify real datasets?
   If this is the case, the implications are huge. You would not need to ask 
   thousands of volunteers for hours of time. Instead, you could easily create
   arbitrary synthetic datasets for your target words.
   

------------------------
Synthetic data creation
------------------------

Here I describe how the synthetic audio samples were created.

1. The list of words is in "inputwords"
2. Pronounciations were taken from the British English Example Pronciation dictionary (BEEP, http://svr-www.eng.cam.ac.uk/comp.speech/Section1/Lexical/beep.html ). The phonemes were translated for the next step with a translation table (see compile.py for details). 
   This creates the file "words". There are multiple pronounciations and stresses for each word.
3. A text-to-speech program (espeak) was used to pronounce these words (see generatetfspeech.sh for details). The pronounciation, stress, pitch, speed and speaker were varied. This gives >1000 clean examples for each word.
4. Noise samples were obtained. 
   Noise samples (airport babble car exhibition restaurant street subway train) come from 
   AURORA (https://www.ee.columbia.edu/~dpwe/sounds/noise/), and additional noise samples were
   synthetically created (ocean white brown pink). (see ../generatenoise.sh for details)
5. Noise and speech were mixed. The speech volume and offset were varied. The noise source, volume was also varied. See addnoise.py for details.  addnoise2.py is the same, but with lower speech volume and higher noise volume.
6. Finally, the data was compressed into an archive and uploaded to kaggle.


------------------------
Acknowledgements
------------------------

This work built upon

* Pronounciation dictionary: BEEP: http://svr-www.eng.cam.ac.uk/comp.speech/Section1/Lexical/beep.html 
* Noise samples: AURORA: https://www.ee.columbia.edu/~dpwe/sounds/noise/ 
* eSPEAK: http://espeak.sourceforge.net/ and mbrola voices http://www.tcts.fpms.ac.be/synthesis/mbrola/mbrcopybin.html

Please provide appropriate citations to the above when using this work.

To cite the resulting dataset, you can use:

APA-style citation: "Buchner J. Synthetic Speech Commands: A public dataset for single-word speech recognition, 2017. Available from <url>".

BibTeX @article{speechcommands, title={Synthetic Speech Commands: A public dataset for single-word speech recognition.}, author={Buchner, Johannes}, journal={Dataset available from <url>}, year={2017} }



