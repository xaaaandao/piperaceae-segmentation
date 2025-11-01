import click
import cv2
import datetime
import math
import numpy as np
import os
import pathlib
# import sklearn.model_selection
import sys
import tensorflow as tf
import time

from albumentations import (
    Compose, HorizontalFlip, Affine, ElasticTransform,
    RandomBrightnessContrast, RandomGamma
)
from AugmentationSequence import AugmentationSequence
from sklearn.model_selection import KFold, train_test_split

# from image import save_figs, save_lossgraph
from metrics import dice_coef, jaccard_distance
from model import evaluate, unet_model
from save import save


class CreateSequence(tf.keras.utils.Sequence):
    def __init__(self, x_set, y_set, batch_size):
        self.x, self.y = x_set, y_set
        self.batch_size = batch_size

    def __len__(self):
        return int(np.ceil(len(self.x) / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_x = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]

        return batch_x, batch_y


def load_files(channel, input_images, input_masks):
    labels = []
    images = []
    images_name = []
    for p in sorted(pathlib.Path(input_masks).rglob("*")):
        # normaliza
        mask = tf.keras.preprocessing.image.load_img(p.resolve(), color_mode="grayscale")
        mask = tf.keras.preprocessing.image.img_to_array(mask)
        mask = mask / 255
        labels.append(mask)

        if channel == 1:
            image = tf.keras.preprocessing.image.load_img(os.path.join(input_images, "%s.jpeg" % p.stem),
                                                          color_mode="grayscale")
        else:
            image = tf.keras.preprocessing.image.load_img(os.path.join(input_images, "%s.jpeg" % p.stem))
        image = tf.keras.preprocessing.image.img_to_array(image)
        image = image / 255
        images.append(image)
        images_name.append(p)

    return images, images_name, labels


def get_data_augmentation(batch_size, data_augmentation, x_train, y_train, augment):
    return AugmentationSequence(x_train, y_train, batch_size, augment) if data_augmentation else CreateSequence(x_train, y_train, batch_size)


def train(batch_size, channel, data_aug, epochs, img_size, learning_rate, loss_function, output, x_train, x_val, y_train, y_val):
    augment = Compose([
        HorizontalFlip(),
        Affine(border_mode=cv2.BORDER_CONSTANT),
        ElasticTransform(border_mode=cv2.BORDER_CONSTANT),
        RandomBrightnessContrast(),
        RandomGamma()
    ])

    steps_per_epoch = math.ceil(x_train.shape[0] / batch_size)
    train_generator = get_data_augmentation(batch_size, data_aug, x_train, y_train, augment)
    reduce_learning_rate = tf.keras.callbacks.ReduceLROnPlateau(monitor="loss", factor=0.5, patience=3, verbose=1)
    filename = os.path.join(output, "unet.h5")
    checkpointer = tf.keras.callbacks.ModelCheckpoint(filename, verbose=1, save_best_only=True)
    strategy = tf.distribute.MirroredStrategy()

    with strategy.scope():
        model = unet_model(channel, img_size)
        adam_opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        model.compile(optimizer=adam_opt, loss=loss_function, metrics=[dice_coef, jaccard_distance, tf.keras.metrics.Precision(), tf.keras.metrics.Recall()])

    tf.keras.backend.clear_session()
    return model.fit(train_generator, steps_per_epoch=steps_per_epoch, epochs=epochs, validation_data=(x_val, y_val), callbacks=[checkpointer, reduce_learning_rate]), model


@click.command()
@click.option("--channel", default=3)
@click.option("--data_aug", default=False)
@click.option("--batch_size", default=4)
@click.option("--folds", default=5)
@click.option("--epochs", default=75)
@click.option("--img_size", default=256)
@click.option("--input_images", type=str)
@click.option("--input_masks", type=str)
@click.option("--learning_rate", default=0.001)
@click.option("--random_state", default=1234)
@click.option("--test_size", default=0.2)
@click.option("--val_size", default=0.05)
@click.option("--loss_function", type=click.Choice(["dice", "jaccard"]), default="dice")
@click.option("--output_dir", type=str, default="output")
def main(batch_size, channel, data_aug, epochs, folds, img_size, input_images, input_masks, learning_rate, loss_function, 
output_dir, random_state, test_size, val_size):
    gpus = tf.config.experimental.list_physical_devices("GPU")

    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)


    if len(input_images) == 0:
        raise FileNotFoundError("images not found in %s" % input_images)

    if len(input_masks) == 0:
        raise FileNotFoundError("images not found in %s" % input_masks)

    images, masks, labels = load_files(channel, input_images, input_masks)

    x = np.array(images).reshape((len(images), img_size, img_size, channel))
    y = np.array(labels).reshape((len(labels), img_size, img_size, 1))

    print("X.shape %s" % str(x.shape))
    print("y.shape %s" % str(y.shape))

    kf = KFold(n_splits=folds, shuffle=True, random_state=random_state)
    figs = []
    fits = []
    results = []

    for fold, (train_index, test_index) in enumerate(kf.split(x)):
        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]
        x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=val_size, random_state=random_state)
        
        print("X_train.shape: %s " % str(x_train.shape))
        print("X_val.shape: %s " % str(x_val.shape))
        print("X_test.shape: %s " % str(x_test.shape))

        output = os.path.join(output_dir, "fold-%d" % fold)
        os.makedirs(output, exist_ok=True)

        fit, model = train(batch_size, channel, data_aug, epochs, img_size, learning_rate, loss_function, output, x_train, x_val, y_train, y_val)

        figs.append({"channel": channel, "fold": fold, "masks": masks, "img_size": img_size, "index": test_index, "model": model, "output": output, "x": x})
        fits.append({"fold": fold, "fit": fit})
        results.append(evaluate(fold, model, x_train, x_val, x_test, y_train, y_val, y_test))

    tf.keras.backend.clear_session()
    save(batch_size, channel, data_aug, epochs, figs, fits, folds, img_size, input_images, input_masks, learning_rate, loss_function, output_dir, random_state, results, test_size, val_size)


if __name__ == "__main__":
    main()