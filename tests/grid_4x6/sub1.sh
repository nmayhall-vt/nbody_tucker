#!/bin/bash

for nb in {0..1}; do
	for i in $( ls j12.*.m ); do
		echo Submitting: $i $nb
		../../nbtuck_dens1.py -nb $nb -ju ev -j $i -mit 1 -nr 1 > ${i::-2}.nb$nb.out  
	done
done

for nb in {2..8}; do
	for i in $( ls j12.*.m ); do
		echo Submitting: $i
		../../nbtuck_davidson.py -nb $nb -ju ev -j $i -direct 1 -mit 1 -dmit 40 -nr 1 > ${i::-2}.nb$nb.out  
	done
done

for i in $( ls j12.*.m ); do
	echo Submitting: $i
	../../nbtuck_davidson.py -np 4 4 4 4 4 4 4 4 4 4 4 4 -dmit 30 -ju ev -j j12.00.m -nr 1 -mit 1 > ${i::-2}.exact.out  
done
