# train unet
for size in 32 64 128 400 256 512; do
    python3 main.py --input_images "/mnt/eec07521-c36a-4d2b-9047-0110e7749eae/Nextcloud/Dropbox import/datasets/2025/images/iwssip/images/${size}" --input_masks "/mnt/eec07521-c36a-4d2b-9047-0110e7749eae/Nextcloud/Dropbox import/datasets/2025/images/iwssip/mascara/${size}" --output_dir "/mnt/eec07521-c36a-4d2b-9047-0110e7749eae/Nextcloud/Dropbox import/datasets/2025/images/multimedia/unet/resultados/${size}" --img_size ${size}
done

# for size in 32 64 128 400 512; do
#     python3 resize.py --input_dir "/mnt/eec07521-c36a-4d2b-9047-0110e7749eae/Nextcloud/Dropbox import/datasets/2025/images/Piperaceae/Peperomia/no-scale" --output_dir "/mnt/eec07521-c36a-4d2b-9047-0110e7749eae/Nextcloud/Dropbox import/datasets/2025/images/Piperaceae/Peperomia/scale/${size}/original" --img_size ${size}
# done

# resize iwssip
# for img_size in 256; do
#     # python3 resize.py --input_dir "/mnt/eec07521-c36a-4d2b-9047-0110e7749eae/Nextcloud/Dropbox import/datasets/2025/images/iwssip/images/sem_resize" --output_dir "/mnt/eec07521-c36a-4d2b-9047-0110e7749eae/Nextcloud/Dropbox import/datasets/2025/images/iwssip/images/${img_size}" --img_size ${img_size}
#     python3 resize.py --input_dir "/mnt/eec07521-c36a-4d2b-9047-0110e7749eae/Nextcloud/Dropbox import/datasets/2025/images/iwssip/mascara/sem_resize" --output_dir "/mnt/eec07521-c36a-4d2b-9047-0110e7749eae/Nextcloud/Dropbox import/datasets/2025/images/iwssip/mascara/${img_size}" --img_size ${img_size}
# done