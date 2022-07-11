import cv2
import math
import os
import tensorflow
import time

from albumentations import (
    Compose, HorizontalFlip, ShiftScaleRotate, ElasticTransform,
    RandomBrightness, RandomContrast, RandomGamma
)
from AugmentationSequence import AugmentationSequence
from file import create_file_result_cv, save_fit_history
from metrics import calculate_iou_dice, dice_coef, jaccard_distance_loss, sum_all_metrics
from image import plot_loss_graph, save_all_images


def get_cfg(x_train, y_train, path, cfg, index_cv):
    cfg.update({"steps": math.ceil(x_train.shape[0] / cfg["batch"])})
    augment = Compose([
        HorizontalFlip(),
        ShiftScaleRotate(rotate_limit=45, border_mode=cv2.BORDER_CONSTANT),
        ElasticTransform(border_mode=cv2.BORDER_CONSTANT),
        RandomBrightness(),
        RandomContrast(),
        RandomGamma()
    ])
    unet_filename = get_filename(path, cfg, index_cv)
    cfg_train = {"train_generator": AugmentationSequence(x_train, y_train, cfg["batch"], augment),
                 "reduce_learning_rate": tensorflow.keras.callbacks.ReduceLROnPlateau(monitor="loss", factor=0.5, patience=3,
                                                                        verbose=1),
                 "checkpointer": tensorflow.keras.callbacks.ModelCheckpoint(unet_filename, verbose=1, save_best_only=True),
                 "strategy": tensorflow.distribute.MirroredStrategy()}
    return cfg_train, unet_filename


def get_filename(path, cfg, index_cv):
    filename = f"unet+batch{cfg['batch']}+lr{str(cfg['learning_rate']).replace('.', '_')}+epoch{cfg['epochs']}+steps{cfg['steps']}"
    filename = filename + f"+cv{index_cv}.h5" if index_cv else ".h5"
    return os.path.join(path, filename)


def get_unet_model(keras=None, img_size=None):
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


def train_test(x_train, y_train, x_val, y_val, x_test, y_test, path, cfg, index_cv=None, sum_metrics=None):
    if index_cv:
        print(f"cv -> {index_cv}")
    print(f"x_train.shape: {x_train.shape}, y_train.shape: {y_train.shape}")
    print(f"x_val.shape: {x_val.shape}, y_val.shape: {y_val.shape}")
    print(f"x_test.shape: {x_test.shape}, y_test.shape: {y_test.shape}")

    # set_idx(x_test)
    cfg_train, unet_filename = get_cfg(x_train, y_train, path, cfg, index_cv)

    elapsed_time = None
    model = None
    fit = None

    if os.path.exists(unet_filename):
        model = tensorflow.keras.models.load_model(unet_filename,
                                                   custom_objects={"jaccard_distance_loss": jaccard_distance_loss,
                                                                   "dice_coef": dice_coef})
    else:
        elapsed_time, fit, model = train(x_val, y_val, cfg, cfg_train)
        save_fit_history(fit, index_cv, path)
        plot_loss_graph(fit, index_cv, path)

    test(x_test, x_train, x_val, y_test, y_train, y_val, cfg, elapsed_time, index_cv, model, path, sum_metrics,
         unet_filename)


def test(x_test, x_train, x_val, y_test, y_train, y_val, cfg, elapsed_time, index_cv, model, path, sum_metrics,
         unet_filename):
    metrics = calculate_iou_dice(x_test, x_train, x_val, y_test, y_train, y_val, model)

    if sum_metrics:
        sum_all_metrics(sum_metrics, metrics)

    create_file_result_cv(x_test, x_train, x_val, cfg, metrics, index_cv, path, unet_filename, elapsed_time)
    save_all_images(x_test, x_train, x_val, y_test, y_train, y_val, model, path)


def train(x_val, y_val, cfg, cfg_train):
    with cfg_train["strategy"].scope():
        model = get_unet_model()
        adam_opt = tensorflow.keras.optimizers.Adam(learning_rate=cfg["learning_rate"])
        model.compile(optimizer=adam_opt, loss=jaccard_distance_loss, metrics=[dice_coef])

    start_time = time.time()
    tensorflow.keras.backend.clear_session()
    fit = model.fit(cfg_train["train_generator"],
                    steps_per_epoch=cfg["steps"],
                    epochs=cfg["epochs"],
                    validation_data=(x_val, y_val),
                    callbacks=[cfg_train["checkpointer"], cfg_train["reduce_learning_rate"]]
                    )
    elapsed_time = time.time() - start_time
    tensorflow.keras.backend.clear_session()
    print(f"time elapsed {time.strftime('%H:%M:%S', time.gmtime(elapsed_time))}")

    return elapsed_time, fit, model
