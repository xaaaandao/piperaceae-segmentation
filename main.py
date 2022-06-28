import numpy
import pathlib
import sklearn
import sklearn.utils
import skimage
import skimage.io

import cv2

img_size = 300
dir_mask = "images/mascara"
dir_images_resize = f"images/{img_size}x{img_size}/originais/resize"


def return_all_files_of_dir(dir):
    return sorted(pathlib.Path(dir).rglob("*"))


def return_genus(filename):
    list_genus = list(["manekia", "ottonia", "piper", "pothomorphe", "peperomia"])
    return next((genus for genus in list_genus if genus in filename), None)


def load_mask(filename):
    return numpy.float32(skimage.io.imread(filename) / 255)


def return_all_mask():
    return [{"filename": filename.name, "file": load_mask(filename), "genus": return_genus(filename.name)}
            for filename in return_all_files_of_dir(dir_mask)]


def load_original_image(filename):
    return skimage.img_as_float32(skimage.io.imread(filename))


def return_all_original_images():
    return [{"filename": filename.name, "file": load_original_image(filename), "genus": return_genus(filename.name)}
            for filename in return_all_files_of_dir(dir_images_resize)]


def main():
    list_all_mask = return_all_mask()
    list_all_original_images = return_all_original_images()

    list_only_images = [image.get("file") for image in list_all_original_images]
    list_only_genus = [image.get("genus") for image in list_all_original_images]

    print(f"len(list_only_images) {len(list_only_images)}")
    print(f"len(list_only_genus) {len(list_only_genus)}")

    X = numpy.array(list_only_images).reshape((len(list_only_images), img_size, img_size, 1))
    Y = numpy.array(list_only_genus).reshape((len(list_only_genus), img_size, img_size, 1))
    X, Y = sklearn.utils.shuffle(X, Y, random_state=1234)

    X_train, X_test, Y_train, Y_test = sklearn.train_test_split(X, Y, test_size=0.05, random_state=1234)
    X_train, X_val, Y_train, Y_val = sklearn.train_test_split(X_train, Y_train, test_size=0.05, random_state=1234)

    print(X_train.shape)
    print(X_val.shape)
    print(X_test.shape)
    print(X.shape)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
