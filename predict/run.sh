#!/bin/bash

PY=~/miniconda3/bin/python

echo "nao esqueca de mudar o diretorio"
$PY predict.py -c RGB -s 256 -t genus --threshold 10