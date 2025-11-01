# piperaceae-segmentation

This repository contains U-Net and was related to this work.
Exsiccate had a background removed using U-Net.

## Source code
- `predict`: contains a Python code to generate a mask.
	- **WARNING**: one of the inputs must be a U-Net trained.
	- **WARNING**: the inputs must be square images.
- `results`: merge results in one file.
- `main.py`: 
	- train the U-Net with the original image and handcraft mask.
	- contains the learning rate, batch, and epochs.
	- U-Net is evaluated using two metrics: the Jaccard Index and the Dice-Sørensen coefficient;
	- **WARNING**: the inputs must be square images.
