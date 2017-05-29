#!/bin/bash

for nb in {0..3}; do
	for i in $( ls j12.*.m ); do
		echo Submitting: $i
		../../nbtuck_davidson.py -nb $nb -ju ev -j $i -direct 0 -mit 1 -dmit 40 -nr 1 -pt 2 > ${i::-2}.nb$nb.pt2.out  & 
		../../nbtuck_davidson.py -nb $nb -ju ev -j $i -direct 0 -mit 1 -dmit 40 -nr 1 -pt 2 -pt_type en > ${i::-2}.nb$nb.pt2.en.out  
	done
done

for nb in {4..4}; do
	for i in $( ls j12.*.m ); do
		echo Submitting: $i
		../../nbtuck_davidson.py -nb $nb -ju ev -j $i -direct 1 -mit 1 -dmit 40 -nr 1 -pt 2 > ${i::-2}.nb$nb.pt2.out  
		../../nbtuck_davidson.py -nb $nb -ju ev -j $i -direct 1 -mit 1 -dmit 40 -nr 1 -pt 2 -pt_type en > ${i::-2}.nb$nb.pt2.en.out  
	done
done
