import numpy as np
import os
import PIL
import random
import tensorflow

from IPython.display import Image, display
from PIL import ImageOps
# from tensorflow import keras
# from tensorflow.keras import tensorflow.keras.layers
# from tensorflow.keras.preprocessing.image import tensorflow.keras.preprocessing.image.load_img


class OxfordPets(tensorflow.keras.utils.Sequence):
    """Helper to iterate over the data (as Numpy arrays)."""

    def __init__(self, batch_size, img_size, input_img_paths, target_img_paths):
        self.batch_size = batch_size
        self.img_size = img_size
        self.input_img_paths = input_img_paths
        self.target_img_paths = target_img_paths

    def __len__(self):
        return len(self.target_img_paths) // self.batch_size

    def __getitem__(self, idx):
        """Returns tuple (input, target) correspond to batch #idx."""
        i = idx * self.batch_size
        batch_input_img_paths = self.input_img_paths[i : i + self.batch_size]
        batch_target_img_paths = self.target_img_paths[i : i + self.batch_size]
        x = np.zeros((self.batch_size,) + self.img_size + (3,), dtype="float32")
        for j, path in enumerate(batch_input_img_paths):
            img = tensorflow.keras.preprocessing.image.load_img(path, target_size=self.img_size)
            x[j] = img
        y = np.zeros((self.batch_size,) + self.img_size + (1,), dtype="uint8")
        for j, path in enumerate(batch_target_img_paths):
            img = tensorflow.keras.preprocessing.image.load_img(path, target_size=self.img_size, color_mode="grayscale")
            y[j] = np.expand_dims(img, 2)
            # Ground truth labels are 1, 2, 3. Subtract one to make them 0, 1, 2:
            y[j] -= 1
        return x, y


def get_model(img_size, num_classes):
    inputs = tensorflow.keras.Input(shape=img_size + (3,))

    ### [First half of the network: downsampling inputs] ###

    # Entry block
    x = tensorflow.keras.layers.Conv2D(32, 3, strides=2, padding="same")(inputs)
    x = tensorflow.keras.layers.BatchNormalization()(x)
    x = tensorflow.keras.layers.Activation("relu")(x)

    previous_block_activation = x  # Set aside residual

    # Blocks 1, 2, 3 are identical apart from the feature depth.
    for filters in [64, 128, 256]:
        x = tensorflow.keras.layers.Activation("relu")(x)
        x = tensorflow.keras.layers.SeparableConv2D(filters, 3, padding="same")(x)
        x = tensorflow.keras.layers.BatchNormalization()(x)

        x = tensorflow.keras.layers.Activation("relu")(x)
        x = tensorflow.keras.layers.SeparableConv2D(filters, 3, padding="same")(x)
        x = tensorflow.keras.layers.BatchNormalization()(x)

        x = tensorflow.keras.layers.MaxPooling2D(3, strides=2, padding="same")(x)

        # Project residual
        residual = tensorflow.keras.layers.Conv2D(filters, 1, strides=2, padding="same")(
            previous_block_activation
        )
        x = tensorflow.keras.layers.add([x, residual])  # Add back residual
        previous_block_activation = x  # Set aside next residual

    ### [Second half of the network: upsampling inputs] ###

    for filters in [256, 128, 64, 32]:
        x = tensorflow.keras.layers.Activation("relu")(x)
        x = tensorflow.keras.layers.Conv2DTranspose(filters, 3, padding="same")(x)
        x = tensorflow.keras.layers.BatchNormalization()(x)

        x = tensorflow.keras.layers.Activation("relu")(x)
        x = tensorflow.keras.layers.Conv2DTranspose(filters, 3, padding="same")(x)
        x = tensorflow.keras.layers.BatchNormalization()(x)

        x = tensorflow.keras.layers.UpSampling2D(2)(x)

        # Project residual
        residual = tensorflow.keras.layers.UpSampling2D(2)(previous_block_activation)
        residual = tensorflow.keras.layers.Conv2D(filters, 1, padding="same")(residual)
        x = tensorflow.keras.layers.add([x, residual])  # Add back residual
        previous_block_activation = x  # Set aside next residual

    # Add a per-pixel classification layer
    outputs = tensorflow.keras.layers.Conv2D(num_classes, 3, activation="softmax", padding="same")(x)

    # Define the model
    model = tensorflow.keras.Model(inputs, outputs)
    return model


def main():
    # os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    # gpus = tensorflow.config.experimental.list_physical_devices("GPU")
    # if gpus:
    #     try:
    #         # Currently, memory growth needs to be the same across GPUs
    #         for gpu in gpus:
    #             tensorflow.config.experimental.set_memory_growth(gpu, True)
    #     except RuntimeError as e:
    #         print(e)

    input_dir = "images/"
    target_dir = "annotations/trimaps/"
    img_size = (160, 160)
    num_classes = 3
    batch_size = 32

    input_img_paths = sorted(
        [
            os.path.join(input_dir, fname)
            for fname in os.listdir(input_dir)
            if fname.endswith(".jpg")
        ]
    )
    target_img_paths = sorted(
        [
            os.path.join(target_dir, fname)
            for fname in os.listdir(target_dir)
            if fname.endswith(".png") and not fname.startswith(".")
        ]
    )

    print("Number of samples:", len(input_img_paths))

    for input_path, target_path in zip(input_img_paths[:10], target_img_paths[:10]):
        print(input_path, "|", target_path)

    # Free up RAM in case the model definition cells were run multiple times
    tensorflow.keras.backend.clear_session()

    # Build model
    model = get_model(img_size, num_classes)
    model.summary()

    # Split our img paths into a training and a validation set
    val_samples = 1000
    random.Random(1337).shuffle(input_img_paths)
    random.Random(1337).shuffle(target_img_paths)
    train_input_img_paths = input_img_paths[:-val_samples]
    train_target_img_paths = target_img_paths[:-val_samples]
    val_input_img_paths = input_img_paths[-val_samples:]
    val_target_img_paths = target_img_paths[-val_samples:]

    # Instantiate data Sequences for each split
    train_gen = OxfordPets(
        batch_size, img_size, train_input_img_paths, train_target_img_paths
    )
    val_gen = OxfordPets(batch_size, img_size, val_input_img_paths, val_target_img_paths)


    # Configure the model for training.
    # We use the "sparse" version of categorical_crossentropy
    # because our target data is integers.
    model.compile(optimizer="rmsprop", loss="sparse_categorical_crossentropy")

    callbacks = [
        tensorflow.keras.callbacks.ModelCheckpoint("oxford_segmentation.h5", save_best_only=True)
    ]

    # Train the model, doing validation at the end of each epoch.
    epochs = 15
    model.fit(train_gen, epochs=epochs, validation_data=val_gen, callbacks=callbacks)

    val_gen = OxfordPets(batch_size, img_size, val_input_img_paths, val_target_img_paths)
    val_preds = model.predict(val_gen)

    def display_mask(i):
        """Quick utility to display a model's prediction."""
        mask = np.argmax(val_preds[i], axis=-1)
        mask = np.expand_dims(mask, axis=-1)
        img = PIL.ImageOps.autocontrast(tensorflow.keras.preprocessing.image.array_to_img(mask))
        display(img)

    # Display results for validation image #10
    i = 10

    # Display input image
    display(Image(filename=val_input_img_paths[i]))

    # Display ground-truth target mask
    img = PIL.ImageOps.autocontrast(tensorflow.keras.preprocessing.image.load_img(val_target_img_paths[i]))
    display(img)

    # Display mask predicted by our model
    display_mask(i)  # Note that the model only sees inputs at 150x150.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()