import tensorflow
import time


def evaluate(end_time, fold, model, x_train, x_val, x_test, y_train, y_val, y_test):
    loss_train, dice_train, jaccard_train, precision_train, recall_train = \
        model.evaluate(x_train, y_train, verbose=False)
    loss_val, dice_val, jaccard_val, precision_val, recall_val = \
        model.evaluate(x_val, y_val, verbose=False)
    loss_test, dice_test, jaccard_test, precision_test, recall_test = \
        model.evaluate(x_test, y_test, verbose=False)
    return {
        "fold": fold,
        "time": str(time.strftime("%H:%M:%S", time.gmtime(end_time))),
        "loss_train": loss_train,
        "dice_train": dice_train,
        "jaccard_train": jaccard_train,
        "precision_train": precision_train,
        "recall_train": recall_train,
        "loss_val": loss_val,
        "dice_val": dice_val,
        "jaccard_val": jaccard_val,
        "precision_val": precision_val,
        "recall_val": recall_val,
        "loss_test": loss_test,
        "dice_test": dice_test,
        "jaccard_test": jaccard_test,
        "precision_test": precision_test,
        "recall_test": recall_test
    }


def unet_model(cfg):
    input_img = tensorflow.keras.layers.Input((cfg["image_size"], cfg["image_size"], cfg["channel"]), name="img")

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
