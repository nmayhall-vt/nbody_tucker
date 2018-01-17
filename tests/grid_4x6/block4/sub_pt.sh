#!/bin/bash


for nb in {0..2}; do
	for i in $( ls j12.*.m ); do
		echo Submitting: $i
		../../../nbtuck_davidson.py -nb $nb -ju ev -j $i -dmit 40 -nr 1 -pt 2 > ${i::-2}.nb$nb.pt.out  
	done
done

