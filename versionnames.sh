#!/bin/bash
for i in 0 1 2 3 4
do
	for i in english english-north en-scottish english_rp english_wmids english-us en-westindies mb-us1 mb-us2
	do 
		for k in 0 10 20 30 40 50 60 70 80 90 99
		do
			for j in 80 100 120 140 160; do 
				echo $versionid $i $j $k
				((versionid++))
			done
		done
	done
done




