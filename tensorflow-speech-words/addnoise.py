import sys
import random
import subprocess
random.seed(1)

noises = []
for line in open('db.noise.lengths'):
	filename, length = line.split()
	noisesource = filename.replace('db.noise', '').replace('.wav', '')
	if noisesource in ['restaurant', 'street', 'white', 'street', 'babble']:
		noisevolumeceil = 0.2
	else:
		noisevolumeceil = 0.5
		#noisevolume = random.normalvariate(1, 0.2)
	
	for noisevolumerel in [0.1, 0.3, 0.5]:
		noisevolume = noisevolumerel * noisevolumeceil
		noises.append((filename, float(length), noisevolume))

def runcmd(cmd):
	#print ' '.join(cmd)
	subprocess.check_call(cmd)

for line in open('db.tfspeech.clean.lengths'):
	filename, length = line.split()
	outfilename = filename.replace('db.tfspeech.clean', 'db.tfspeech.noisy') + '.wav'
	length = float(length)
	if length > 1.2: 
		continue
	length1 = min(length, 1.)
	offset = random.uniform(0, 1-length1)
	total = offset + length
	padend = 0
	if total < 1:
		padend = 1 - total
	
	#print 'sox', filename, 'test.wav', 'pad', offset, padend
	sourcevol = random.uniform(0.1, 0.5)
	noisefile, noiselength, noisevolume = random.choice(noises)
	noisestart = random.uniform(0, noiselength-1)
	#print 'combine', filename, 'offset', offset, padend, 'with noise', noisefile, noisevolume, 'starting at', noisestart
	print filename, noisefile
	cmdsource = ['sox', filename,  '-r', '16000', 'source.wav', 'vol', str(sourcevol), 'pad', '%.2f' % offset, '%.2f' % padend]
	cmdnoise  = ['sox', noisefile, '-r', '16000', 'noise.wav', 'vol', str(noisevolume), 'trim', '%.2f' % noisestart, '1.1']
	cmdtotal  = ['sox', '-m', 'source.wav', 'noise.wav', outfilename, 'trim', '0', '1']
	
	runcmd(cmdsource)
	runcmd(cmdnoise)
	runcmd(cmdtotal)

	
	
