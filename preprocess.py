import os
import pathlib

import cv2

# equalize image
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
dir_clahe = "images/400x400/mask/clahe"
dir_mask = "images/mascara"
dir_original = "images/originais"
dir_resize = "images/400x400/mask/resize"
dir_resize_original = "images/400x400/resize"
img_size = 300


def resize_image(image):
    return cv2.resize(image, (img_size, img_size), interpolation=cv2.INTER_CUBIC)


def return_all_mask():
    return [{"filename": file.name, "file": cv2.imread(str(file.resolve()), cv2.IMREAD_GRAYSCALE)}
            for file in pathlib.Path(dir_mask).rglob("*")]

def return_all_original():
    return [{"filename": file.name, "file": cv2.imread(str(file.resolve()), cv2.IMREAD_GRAYSCALE)}
            for file in pathlib.Path(dir_original).rglob("*")]


def create_if_not_exists_dir(dir):
    if not os.path.isdir(dir):
        os.makedirs(dir)


def apply_clahe(list_mask):
    create_if_not_exists_dir(dir_clahe)
    for mask in list_mask:
        cv2.imwrite(os.path.join(dir_clahe, mask["filename"]), resize_image(clahe.apply(mask["file"])))


def only_resize(list_mask):
    create_if_not_exists_dir(dir_resize_original)
    for mask in list_mask:
        cv2.imwrite(os.path.join(dir_resize_original, mask["filename"]), resize_image(mask["file"]))


def main():
    # list_mask = return_all_mask()
    # apply_clahe(list_mask)
    # only_resize(list_mask)

    list_original = return_all_original()
    only_resize(list_original)


if __name__ == '__main__':
    main()