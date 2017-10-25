import sys

phonemedict = {}

print 'reading phoneme translation table...'
for row in open('phoncodemod.doc'):
	if row.strip() == '': continue
	ARPAbet, MRPA, Edin, Alvey = row.strip().split()[:4]
	phonemedict[ARPAbet] = Alvey
	phonemedict[ARPAbet.lower()] = Alvey

print 'reading words...'
words = [word.strip().lower() for word in open('onesyllablewords')]
words_set = set(words)

out = open('words', 'w')
print 'finding phonemes...'
for row in open('beep-1.0'):
	if row.startswith('#'): continue
	parts = row.strip().split()
	if len(parts) == 0: 
		continue
	word = parts[0].lower()
	if word not in words_set:
		continue
	wordid = words.index(word)
	phonemes = parts[1:]
	phonemes = [phonemedict[phoneme.lower()] for phoneme in phonemes]
	for i in range(len(phonemes)+1):
		stressed_phonemes = phonemes[:i] + ["'"] + phonemes[i:]
		out.write('%s %d %s\n' % (word, wordid, '[[%s]]' % ''.join(stressed_phonemes)))
		
	



