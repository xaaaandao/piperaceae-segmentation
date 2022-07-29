import cv2
import math
import os
import tensorflow
import time

from AugmentationSequence import AugmentationSequence
from albumentations import (
    Compose, HorizontalFlip, ShiftScaleRotate, ElasticTransform,
    RandomBrightness, RandomContrast, RandomGamma,
)
from metrics import dice_loss, jaccard_distance_loss
from files import save_fit_history, save_lossgraph


def get_filename(cfg, fold, path, steps):
    filename = f"batch{cfg['batch_size']}+lr{str(cfg['learning_rate']).replace('.', '_')}+epoch{str(cfg['epochs'])}+steps{steps}+fold{fold}+unet.h5"
    return os.path.join(path, filename)


def cfg_model(cfg, fold, path, x_train, y_train):
    steps_per_epoch = math.ceil(x_train.shape[0] / cfg["batch_size"])
    augment = Compose([
        HorizontalFlip(),
        ShiftScaleRotate(rotate_limit=45, border_mode=cv2.BORDER_CONSTANT),
        ElasticTransform(border_mode=cv2.BORDER_CONSTANT),
        RandomBrightness(),
        RandomContrast(),
        RandomGamma()
    ])
    train_generator = AugmentationSequence(x_train, y_train, cfg["batch_size"], augment)
    model_filename = get_filename(cfg, fold, path, steps_per_epoch)
    reduce_learning_rate = tensorflow.keras.callbacks.ReduceLROnPlateau(monitor="loss", factor=0.5, patience=3,
                                                                        verbose=1)
    checkpointer = tensorflow.keras.callbacks.ModelCheckpoint(model_filename, verbose=1, save_best_only=True)
    strategy = tensorflow.distribute.MirroredStrategy()
    return checkpointer, reduce_learning_rate, steps_per_epoch, strategy, train_generator, model_filename


def train_model(cfg, checkpointer, fold, path, reduce_learning_rate, steps_per_epoch, strategy, train_generator, x_val, y_val):
    with strategy.scope():
        model = unet_model()
        adam_opt = tensorflow.keras.optimizers.Adam(learning_rate=cfg["learning_rate"])
        model.compile(optimizer=adam_opt, loss=jaccard_distance_loss, metrics=[dice_loss])

    tensorflow.keras.backend.clear_session()

    start_time = time.time()
    fit = model.fit(train_generator,
                    steps_per_epoch=steps_per_epoch,
                    epochs=cfg["epochs"],
                    validation_data=(x_val, y_val),
                    callbacks=[checkpointer, reduce_learning_rate]
                    )
    elapsed_time = time.time() - start_time
    print(f"time elapsed {time.strftime('%H:%M:%S', time.gmtime(elapsed_time))}")

    save_fit_history(fold, fit, path)
    save_lossgraph(fold, fit, path)

    return elapsed_time, model


def unet_model(keras=None, img_size=None):
    input_img = tensorflow.keras.layers.Input((img_size, img_size, 1), name="img")

    # Contract #1
    c1 = tensorflow.keras.layers.Conv2D(16, (3, 3), kernel_initializer="he_uniform", padding="same")(input_img)
    c1 = tensorflow.keras.layers.BatchNormalization()(c1)
    c1 = tensorflow.keras.layers.Activation("relu")(c1)
    c1 = tensorflow.keras.layers.Dropout(0.1)(c1)
    c1 = tensorflow.keras.layers.Conv2D(16, (3, 3), kernel_initializer="he_uniform", padding="same")(c1)
    c1 = tensorflow.keras.layers.BatchNormalization()(c1)
    c1 = tensorflow.keras.layers.Activation("relu")(c1)
    p1 = tensorflow.keras.layers.MaxPooling2D((2, 2))(c1)

    # Contract #2
    c2 = tensorflow.keras.layers.Conv2D(32, (3, 3), kernel_initializer="he_uniform", padding="same")(p1)
    c2 = tensorflow.keras.layers.BatchNormalization()(c2)
    c2 = tensorflow.keras.layers.Activation("relu")(c2)
    c2 = tensorflow.keras.layers.Dropout(0.2)(c2)
    c2 = tensorflow.keras.layers.Conv2D(32, (3, 3), kernel_initializer="he_uniform", padding="same")(c2)
    c2 = tensorflow.keras.layers.BatchNormalization()(c2)
    c2 = tensorflow.keras.layers.Activation("relu")(c2)
    p2 = tensorflow.keras.layers.MaxPooling2D((2, 2))(c2)

    # Contract #3
    c3 = tensorflow.keras.layers.Conv2D(64, (3, 3), kernel_initializer="he_uniform", padding="same")(p2)
    c3 = tensorflow.keras.layers.BatchNormalization()(c3)
    c3 = tensorflow.keras.layers.Activation("relu")(c3)
    c3 = tensorflow.keras.layers.Dropout(0.3)(c3)
    c3 = tensorflow.keras.layers.Conv2D(64, (3, 3), kernel_initializer="he_uniform", padding="same")(c3)
    c3 = tensorflow.keras.layers.BatchNormalization()(c3)
    c3 = tensorflow.keras.layers.Activation("relu")(c3)
    p3 = tensorflow.keras.layers.MaxPooling2D((2, 2))(c3)

    # Contract #4
    c4 = tensorflow.keras.layers.Conv2D(128, (3, 3), kernel_initializer="he_uniform", padding="same")(p3)
    c4 = tensorflow.keras.layers.BatchNormalization()(c4)
    c4 = tensorflow.keras.layers.Activation("relu")(c4)
    c4 = tensorflow.keras.layers.Dropout(0.4)(c4)
    c4 = tensorflow.keras.layers.Conv2D(128, (3, 3), kernel_initializer="he_uniform", padding="same")(c4)
    c4 = tensorflow.keras.layers.BatchNormalization()(c4)
    c4 = tensorflow.keras.layers.Activation("relu")(c4)
    p4 = tensorflow.keras.layers.MaxPooling2D((2, 2))(c4)

    # Middle
    c5 = tensorflow.keras.layers.Conv2D(256, (3, 3), kernel_initializer="he_uniform", padding="same")(p4)
    c5 = tensorflow.keras.layers.BatchNormalization()(c5)
    c5 = tensorflow.keras.layers.Activation("relu")(c5)
    c5 = tensorflow.keras.layers.Dropout(0.5)(c5)
    c5 = tensorflow.keras.layers.Conv2D(256, (3, 3), kernel_initializer="he_uniform", padding="same")(c5)
    c5 = tensorflow.keras.layers.BatchNormalization()(c5)
    c5 = tensorflow.keras.layers.Activation("relu")(c5)

    # Expand (upscale) #1
    u6 = tensorflow.keras.layers.Conv2DTranspose(128, (3, 3), strides=(2, 2), padding="same")(c5)
    u6 = tensorflow.keras.layers.concatenate([u6, c4])
    c6 = tensorflow.keras.layers.Conv2D(128, (3, 3), kernel_initializer="he_uniform", padding="same")(u6)
    c6 = tensorflow.keras.layers.BatchNormalization()(c6)
    c6 = tensorflow.keras.layers.Activation("relu")(c6)
    c6 = tensorflow.keras.layers.Dropout(0.5)(c6)
    c6 = tensorflow.keras.layers.Conv2D(128, (3, 3), kernel_initializer="he_uniform", padding="same")(c6)
    c6 = tensorflow.keras.layers.BatchNormalization()(c6)
    c6 = tensorflow.keras.layers.Activation("relu")(c6)

    # Expand (upscale) #2
    u7 = tensorflow.keras.layers.Conv2DTranspose(64, (3, 3), strides=(2, 2), padding="same")(c6)
    u7 = tensorflow.keras.layers.concatenate([u7, c3])
    c7 = tensorflow.keras.layers.Conv2D(64, (3, 3), kernel_initializer="he_uniform", padding="same")(u7)
    c7 = tensorflow.keras.layers.BatchNormalization()(c7)
    c7 = tensorflow.keras.layers.Activation("relu")(c7)
    c7 = tensorflow.keras.layers.Dropout(0.5)(c7)
    c7 = tensorflow.keras.layers.Conv2D(64, (3, 3), kernel_initializer="he_uniform", padding="same")(c7)
    c7 = tensorflow.keras.layers.BatchNormalization()(c7)
    c7 = tensorflow.keras.layers.Activation("relu")(c7)

    # Expand (upscale) #3
    u8 = tensorflow.keras.layers.Conv2DTranspose(32, (3, 3), strides=(2, 2), padding="same")(c7)
    u8 = tensorflow.keras.layers.concatenate([u8, c2])
    c8 = tensorflow.keras.layers.Conv2D(32, (3, 3), kernel_initializer="he_uniform", padding="same")(u8)
    c8 = tensorflow.keras.layers.BatchNormalization()(c8)
    c8 = tensorflow.keras.layers.Activation("relu")(c8)
    c8 = tensorflow.keras.layers.Dropout(0.5)(c8)
    c8 = tensorflow.keras.layers.Conv2D(32, (3, 3), kernel_initializer="he_uniform", padding="same")(c8)
    c8 = tensorflow.keras.layers.BatchNormalization()(c8)
    c8 = tensorflow.keras.layers.Activation("relu")(c8)

    # Expand (upscale) #4
    u9 = tensorflow.keras.layers.Conv2DTranspose(16, (3, 3), strides=(2, 2), padding="same")(c8)
    u9 = tensorflow.keras.layers.concatenate([u9, c1])
    c9 = tensorflow.keras.layers.Conv2D(16, (3, 3), kernel_initializer="he_uniform", padding="same")(u9)
    c9 = tensorflow.keras.layers.BatchNormalization()(c9)
    c9 = tensorflow.keras.layers.Activation("relu")(c9)
    c9 = tensorflow.keras.layers.Dropout(0.5)(c9)
    c9 = tensorflow.keras.layers.Conv2D(16, (3, 3), kernel_initializer="he_uniform", padding="same")(c9)
    c9 = tensorflow.keras.layers.BatchNormalization()(c9)
    c9 = tensorflow.keras.layers.Activation("relu")(c9)

    output = tensorflow.keras.layers.Conv2D(1, (1, 1), activation="sigmoid")(c9)
    model = tensorflow.keras.Model(inputs=[input_img], outputs=[output])
    return model
