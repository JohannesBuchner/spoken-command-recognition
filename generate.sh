#!/bin/bash
lastwordid=""
cat words | while read word wordid phoneme
do
	echo $word
	mkdir -p db/$word
	if [[ $word != $lastword ]]; then
		versionid=0
	fi
	lastword=$word
	for i in english english-north en-scottish english_rp english_wmids english-us en-westindies mb-us1 mb-us2
	do 
		for k in 0 10 20 30 40 50 60 70 80 90 99
		do
			for j in 80 100 120 140 160; do 
				echo $versionid "$phoneme" $i $j $k
				echo "$phoneme" | espeak -p $k -s $j -v $i -w cleanword.wav
				#sox cleanword.wav cleanwordtrimmed.wav silence -l 1 0.1 1% -1 2.0 1%
				sox cleanword.wav cleanwordtrimmed.wav silence 1 0.1 0.1% reverse silence 1 0.1 0.1% reverse
				opusenc --quiet --hard-cbr --bitrate 8 cleanwordtrimmed.wav db/$word/$versionid
				((versionid++))
			done
		done
	done
done




