mkdir -p db.noise
pushd db.noise || exit
for i in airport babble car exhibition restaurant street subway train
do
	wget --no-clobber https://www.ee.columbia.edu/~dpwe/sounds/noise/$i.wav 
done

sox -t s16 -r 8000 -c 1 /dev/zero ocean.wav synth 30 brownnoise synth pinknoise mix synth sine amod 0.3 1 vol 0.1
sox -t s16 -r 8000 -c 1 /dev/zero white.wav synth 30 whitenoise vol 0.1 
sox -t s16 -r 8000 -c 1 /dev/zero brown.wav synth 30 brownnoise vol 0.1 
sox -t s16 -r 8000 -c 1 /dev/zero pink.wav  synth 30 pinknoise vol 0.1 

popd
