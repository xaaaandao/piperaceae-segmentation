import os
import pathlib

import cv2

# equalize image
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
# dir_clahe = "images/clahe"
# dir_resize = "images/resize"
output_dir = "pimentas/originais_resize"
dir_mask = "pimentas/originais"
img_size = 400


def resize_image(image):
    return cv2.resize(image, (img_size, img_size), interpolation=cv2.INTER_CUBIC)


def return_all_mask():
    return [{"filename": file.name, "file": cv2.imread(str(file.resolve()), cv2.IMREAD_GRAYSCALE)}
            for file in pathlib.Path(dir_mask).rglob("*")]


def create_if_not_exists_dir(dir):
    if not os.path.isdir(dir):
        os.makedirs(dir)




# def apply_clahe(list_mask):
#     create_if_not_exists_dir(dir_clahe)
#     for mask in list_mask:
#         cv2.imwrite(os.path.join(dir_clahe, mask["filename"]), resize_image(clahe.apply(mask["file"])))
#
#


def only_resize(list_mask):
    create_if_not_exists_dir(output_dir)
    for mask in list_mask:
        cv2.imwrite(os.path.join(output_dir, mask["filename"]), resize_image(mask["file"]))


def main():
    list_mask = return_all_mask()
    # apply_clahe(list_mask)
    only_resize(list_mask)


if __name__ == '__main__':
    main()