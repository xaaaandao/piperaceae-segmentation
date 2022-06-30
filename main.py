import sys

import AugmentationSequence
import cv2
import datetime
import math
import numpy
import os
import pathlib
import sklearn.utils
import sklearn.model_selection
import skimage
import skimage.io
import tensorflow

from albumentations import (
    Compose, HorizontalFlip, ShiftScaleRotate, ElasticTransform,
    RandomBrightness, RandomContrast, RandomGamma
)

import config
import unet


def carrega_mascara(nome_arquivo):
    return numpy.float32(skimage.io.imread(nome_arquivo) / 255)


def retorna_todas_mascaras():
    return [carrega_mascara(nome_arquivo) for nome_arquivo in sorted(pathlib.Path("pimentas/resize").rglob("*"))]


def carrega_imagem_original(nome_arquivo):
    return skimage.img_as_float32(skimage.io.imread(nome_arquivo))


def retorna_todas_imgs_originais():
    return [carrega_imagem_original(nome_arquivo) for nome_arquivo in sorted(pathlib.Path("pimentas/originais_resize").rglob("*"))]


# LOSS Functions
def jaccard_distance_loss(y_true, y_pred, smooth=100):
    intersection = tensorflow.keras.backend.sum(tensorflow.keras.backend.abs(y_true * y_pred), axis=-1)
    union = tensorflow.keras.backend.sum(tensorflow.keras.backend.abs(y_true) + tensorflow.keras.backend.abs(y_pred), axis=-1)
    jac = (intersection + smooth) / (union - intersection + smooth)
    loss = (1 - jac) * smooth
    return loss


def dice_coef(y_true, y_pred, smooth=1):
    intersection = tensorflow.keras.backend.sum(tensorflow.keras.backend.abs(y_true * y_pred), axis=-1)
    union = tensorflow.keras.backend.sum(tensorflow.keras.backend.abs(y_true), -1) + tensorflow.keras.backend.sum(tensorflow.keras.backend.abs(y_pred), -1)
    return (2. * intersection + smooth) / (union + smooth)


def main():
    config.define_avx_avx2()
    config.define_gpu()

    img_size = 400
    lista_todas_mascaras = retorna_todas_mascaras()
    lista_todas_imgs_originais = retorna_todas_imgs_originais()

    X = numpy.array(lista_todas_imgs_originais).reshape(len(lista_todas_imgs_originais), img_size, img_size, 1)
    Y = numpy.array(lista_todas_mascaras).reshape(len(lista_todas_mascaras), img_size, img_size, 1)
    X, Y = sklearn.utils.shuffle(X, Y, random_state=1234)

    X_train, X_test, Y_train, Y_test = sklearn.model_selection.train_test_split(X, Y, test_size=0.05, random_state=1234)
    X_train, X_val, Y_train, Y_val = sklearn.model_selection.train_test_split(X_train, Y_train, test_size=0.05, random_state=1234)

    print(f"X_train.shape: {X_train.shape}")
    print(f"X_val.shape: {X_val.shape}")
    print(f"X_test.shape: {X_test.shape}")
    print(f"X.shape: {X.shape}")

    # adiciona um id para cada imagem de teste???
    lista_todas_imgs_originais_ids = []

    nimages = X_test.shape[0]
    for idx in range(nimages):
        test_image = X_test[idx, :, :, 0]
        if any(numpy.array_equal(test_image, x) for x in lista_todas_imgs_originais):
            lista_todas_imgs_originais_ids.append(idx)

    augment = Compose([
        HorizontalFlip(),
        ShiftScaleRotate(rotate_limit=45, border_mode=cv2.BORDER_CONSTANT),
        ElasticTransform(border_mode=cv2.BORDER_CONSTANT),
        RandomBrightness(),
        RandomContrast(),
        RandomGamma()
    ])

    batch_size = 16
    train_generator = AugmentationSequence.AugmentationSequence(X_train, Y_train, batch_size, augment)
    steps_per_epoch = math.ceil(X_train.shape[0] / batch_size)

    reduce_learning_rate = tensorflow.keras.callbacks.ReduceLROnPlateau(
        monitor="loss",
        factor=0.5,
        patience=3,
        verbose=1
    )

    checkpointer = tensorflow.keras.callbacks.ModelCheckpoint(
        "unet.h5",
        verbose=1,
        save_best_only=True
    )

    strategy = tensorflow.distribute.MirroredStrategy()


    if (os.path.exists("unet.h5")):
        model = tensorflow.keras.models.load_model("unet.h5", custom_objects={"jaccard_distance_loss": jaccard_distance_loss, "dice_coef": dice_coef})
    #
    else:
        with strategy.scope():
            model = unet.unet_model()
            adam_opt = tensorflow.keras.optimizers.Adam(learning_rate=0.001)
            model.compile(optimizer=adam_opt, loss=jaccard_distance_loss, metrics=[dice_coef])
            print(f"-> inicio: {datetime.datetime.now().strftime('%m/%d/%Y, %H:%M:%S')}")

            fit = model.fit(train_generator, steps_per_epoch=steps_per_epoch,
                        epochs=100,
                        validation_data=(X_val, Y_val),
                        callbacks=[
                            checkpointer,
                            reduce_learning_rate])

            print(f"-> fim: {datetime.datetime.now().strftime('%m/%d/%Y, %H:%M:%S')}")

    iou_train, dice_train = model.evaluate(X_train, Y_train, verbose=True)
    iou_val, dice_val = model.evaluate(X_val, Y_val, verbose=True)
    iou_test, dice_test = model.evaluate(X_test, Y_test, verbose=True)

    print("Jaccard distance (IoU) train: %f" % iou_train)
    print("Dice coeffient train: %f" % dice_train)
    print("Jaccard distance (IoU) validation: %f" % iou_val)
    print("Dice coeffient validation: %f" % dice_val)
    print("Jaccard distance (IoU) test: %f" % iou_test)
    print("Dice coeffient test: %f" % dice_test)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
