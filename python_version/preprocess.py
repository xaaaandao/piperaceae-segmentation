import cv2
import os
import pathlib

from file import create_dir_if_not_exists


def resize_image(image, image_size):
    return cv2.resize(image, (image_size, image_size), interpolation=cv2.INTER_CUBIC)


def get_all_images(dir):
    return [{"filename": file.name, "file": cv2.imread(str(file.resolve()), cv2.IMREAD_GRAYSCALE)}
            for file in pathlib.Path(dir).rglob("*")]


def only_resize(list_images, output_dir, image_size):
    create_dir_if_not_exists(output_dir)
    for image in list_images:
        cv2.imwrite(os.path.join(output_dir, image["filename"]), resize_image(image["file"], image_size))


def resize_all():
    for data in [{"path": "new/images", "type": "images"}, {"path": "new/mask", "type": "mask"}]:
        only_resize(get_all_images(data["path"]), data["type"], 400)
