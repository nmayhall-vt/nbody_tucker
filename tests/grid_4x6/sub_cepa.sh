#!/bin/bash

for nb in {2..2}; do
	for i in $( ls j12.*.m ); do
		echo Submitting: $i
		../../nbtuck_davidson.py -nb $nb -ju ev -j $i -direct 0 -mit 1 -dmit 40 -nr 1 > ${i::-2}.nb$nb.cepa.out  & 
	done
done

