#!/bin/bash

for img_size in 32 64 128 256 400 512; do
	python3 predict.py --best "/mnt/eec07521-c36a-4d2b-9047-0110e7749eae/Nextcloud/Dropbox import/datasets/2025/images/iwssip/unet/resultados/${img_size}/best.h5" --input_dir "/mnt/eec07521-c36a-4d2b-9047-0110e7749eae/Nextcloud/Dropbox import/datasets/2025/images/Piperaceae/Peperomia/scale/${img_size}/original/" --output_dir "/mnt/eec07521-c36a-4d2b-9047-0110e7749eae/Nextcloud/Dropbox import/datasets/2025/images/Piperaceae/Peperomia/scale/${img_size}"
done