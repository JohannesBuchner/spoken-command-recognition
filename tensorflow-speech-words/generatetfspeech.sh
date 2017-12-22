#!/bin/bash
lastword=""
cat words | while read word wordid phoneme
do
	echo $word
	mkdir -p db.tfspeech.clean/$word
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
				#[ -e db.tfspeech.clean/$word/$versionid ] && continue
				echo "$phoneme" | espeak -p $k -s $j -v $i --stdout > db.tfspeech.clean/$word/$versionid &
				((versionid++))
			done
			wait
		done
	done
done




