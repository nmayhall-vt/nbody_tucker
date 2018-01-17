#!/bin/bash

for i in $( ls j12.*.m ); do
	echo Submitting: $i
	../../../nbtuck_davidson.py -nb 4 -ju ev -j $i -dmit 40 -nr 1 > ${i::-2}.nb4.out  
done

