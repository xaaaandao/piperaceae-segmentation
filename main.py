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
    Compose, HorizontalFlip, ShiftScaleRotate, ElasticTransform,
    RandomBrightness, RandomContrast, RandomGamma
)
from AugmentationSequence import AugmentationSequence
from sklearn.model_selection import KFold, train_test_split

from files import create_dir, save_fit_history
from image import save_figs, save_lossgraph
from metrics import dice_coef, jaccard_distance
from model import evaluate, unet_model, get_loss_function
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


def load_files(cfg, images_folder, masks_folder):
    list_labels = []
    list_images = []
    list_images_names = []
    for file in sorted(pathlib.Path(masks_folder).rglob('*')):
        mask = tf.keras.preprocessing.image.load_img(file.resolve(), color_mode='grayscale')
        mask = tf.keras.preprocessing.image.img_to_array(mask)
        mask = mask / 255
        list_labels.append(mask)

        if cfg['channel'] == 1:
            image = tf.keras.preprocessing.image.load_img(os.path.join(images_folder, f'{file.stem}.jpeg'),
                                                          color_mode='grayscale')
        else:
            image = tf.keras.preprocessing.image.load_img(os.path.join(images_folder, f'{file.stem}.jpeg'))
        image = tf.keras.preprocessing.image.img_to_array(image)
        image = image / 255
        list_images.append(image)
        list_images_names.append(file)

    return list_images, list_images_names, list_labels


def get_data_augmentation(cfg, x_train, y_train, augment):
    return AugmentationSequence(x_train, y_train, cfg['batch_size'], augment) if cfg['data_augmentation'] else CreateSequence(x_train, y_train, cfg['batch_size'])


def main():
    gpus = tf.config.experimental.list_physical_devices('GPU')

    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                print(f'GPU: {tf.config.experimental.get_device_details(gpu)["device_name"]}')
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)

    cfg = {
        'channel': 1,
        'batch_size': 4,
        'fold': 5,
        'epochs': 75,
        'image_size': 256,
        'learning_rate': 0.001,
        'random_state': 1234,
        'test_size': 0.2,
        'val_size': 0.05,
        'path_dataset': '../dataset',
        'path_out': 'out',
        'loss_function': 'dice',
        'data_augmentation': False
    }

    images_folder = os.path.join('../dataset_gimp', 'imagens_sp', 'imagens', 'grayscale', 'originais', str(cfg['image_size']), 'jpeg')
    masks_folder = os.path.join('../dataset_gimp', 'imagens_sp', 'imagens', 'mask', 'mask_manual', str(cfg['image_size']), 'bmp')

    if len(images_folder) == 0:
        raise FileNotFoundError(f'images not found in {images_folder}')

    if len(masks_folder) == 0:
        raise FileNotFoundError(f'mask not found in {masks_folder}')

    list_images, list_images_names, list_labels = load_files(cfg, images_folder, masks_folder)

    x = np.array(list_images).reshape((len(list_images), cfg['image_size'], cfg['image_size'], cfg['channel']))
    y = np.array(list_labels).reshape((len(list_labels), cfg['image_size'], cfg['image_size'], 1))

    print(x.shape, y.shape)

    kf = KFold(n_splits=cfg['fold'], shuffle=True, random_state=cfg['random_state'])

    models = []
    list_evaluate = []
    list_time = 0
    current_datetime = datetime.datetime.now().strftime('%d-%m-%Y-%H-%M-%S')
    path = os.path.join(cfg['path_out'], current_datetime)
    create_dir([path])

    for fold, (train_index, test_index) in enumerate(kf.split(x)):
        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]
        x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=cfg['val_size'], random_state=cfg['random_state'])

        print(x_train.shape)
        print(x_val.shape)
        print(x_test.shape)
        print(x.shape)

        path_fold = os.path.join(path, str(fold))
        create_dir([path_fold])

        augment = Compose([
            HorizontalFlip(),
            ShiftScaleRotate(rotate_limit=45, border_mode=cv2.BORDER_CONSTANT),
            ElasticTransform(border_mode=cv2.BORDER_CONSTANT),
            RandomBrightness(),
            RandomContrast(),
            RandomGamma()
        ])
        steps_per_epoch = math.ceil(x_train.shape[0] / cfg['batch_size'])
        train_generator = get_data_augmentation(cfg, x_train, y_train, augment)
        reduce_learning_rate = tf.keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.5, patience=3, verbose=1)
        filename_model = os.path.join(path_fold, 'unet.h5')
        checkpointer = tf.keras.callbacks.ModelCheckpoint(filename_model, verbose=1, save_best_only=True)
        strategy = tf.distribute.MirroredStrategy()

        with strategy.scope():
            model = unet_model(cfg)
            adam_opt = tf.keras.optimizers.Adam(learning_rate=cfg['learning_rate'])
            model.compile(optimizer=adam_opt, loss=get_loss_function(cfg['loss_function']), metrics=[dice_coef, jaccard_distance, tf.keras.metrics.Precision(), tf.keras.metrics.Recall()])

        tf.keras.backend.clear_session()
        start_time = time.time()
        fit = model.fit(train_generator, steps_per_epoch=steps_per_epoch, epochs=cfg['epochs'], validation_data=(x_val, y_val), callbacks=[checkpointer, reduce_learning_rate])
        end_time = time.time() - start_time

        save_fit_history(fold, fit, path_fold)
        save_lossgraph(fold, fit, path_fold)
        list_evaluate.append(evaluate(end_time, fold, model, x_train, x_val, x_test, y_train, y_val, y_test))
        list_time += end_time

        models.append(model)
        save_figs(cfg, list_images_names, test_index, model, path_fold, x)

    tf.keras.backend.clear_session()

    save(cfg, sys.argv, images_folder, list_evaluate, list_images, list_labels, list_time, masks_folder, path)


if __name__ == '__main__':
    main()