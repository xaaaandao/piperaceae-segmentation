#!/bin/bash
PY=~/miniconda3/bin/python

echo "nao esqueca de mudar o diretorio"
for color in RGB grayscale;
do
	for image_size in 256 400 512;
	do
		for threshold in 2;
		do
			$PY predict.py -c ${color} -s ${image_size} -t genus --threshold ${threshold}
		done
	done
done
