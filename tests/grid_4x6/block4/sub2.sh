#!/bin/bash
for i in $( ls j12.*.m ); do
	echo Submitting: $i
	../../../nbtuck_davidson.py -np 16 16 16 16 16 16 -dmit 30 -ju ev -j $i -nr 1 -mit 1 > ${i::-2}.exact.out  
done
