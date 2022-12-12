#!/bin/bash
PY=~/miniconda3/bin/python
TAXON=specific_epithet

echo "nao esqueca de mudar o diretorio"
for color in RGB grayscale; do
	for image_size in 256 400 512; do
		${PY} predict.py -c ${color} -s ${image_size} -t ${TAXON}
	done
done
