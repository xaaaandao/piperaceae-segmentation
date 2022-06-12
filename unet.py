
def unet_model(keras=None, img_size=None):

    input_img = keras.layers.Input((img_size, img_size, 1), name = "img")

    # Contract #1
    c1 = keras.layers.Conv2D(16, (3, 3), kernel_initializer = "he_uniform", padding = "same")(input_img)
    c1 = keras.layers.BatchNormalization()(c1)
    c1 = keras.layers.Activation("relu")(c1)
    c1 = keras.layers.Dropout(0.1)(c1)
    c1 = keras.layers.Conv2D(16, (3, 3), kernel_initializer = "he_uniform", padding = "same")(c1)
    c1 = keras.layers.BatchNormalization()(c1)
    c1 = keras.layers.Activation("relu")(c1)
    p1 = keras.layers.MaxPooling2D((2, 2))(c1)

    # Contract #2
    c2 = keras.layers.Conv2D(32, (3, 3), kernel_initializer = "he_uniform", padding = "same")(p1)
    c2 = keras.layers.BatchNormalization()(c2)
    c2 = keras.layers.Activation("relu")(c2)
    c2 = keras.layers.Dropout(0.2)(c2)
    c2 = keras.layers.Conv2D(32, (3, 3), kernel_initializer = "he_uniform", padding = "same")(c2)
    c2 = keras.layers.BatchNormalization()(c2)
    c2 = keras.layers.Activation("relu")(c2)
    p2 = keras.layers.MaxPooling2D((2, 2))(c2)

    # Contract #3
    c3 = keras.layers.Conv2D(64, (3, 3), kernel_initializer = "he_uniform", padding = "same")(p2)
    c3 = keras.layers.BatchNormalization()(c3)
    c3 = keras.layers.Activation("relu")(c3)
    c3 = keras.layers.Dropout(0.3)(c3)
    c3 = keras.layers.Conv2D(64, (3, 3), kernel_initializer = "he_uniform", padding = "same")(c3)
    c3 = keras.layers.BatchNormalization()(c3)
    c3 = keras.layers.Activation("relu")(c3)
    p3 = keras.layers.MaxPooling2D((2, 2))(c3)

    # Contract #4
    c4 = keras.layers.Conv2D(128, (3, 3), kernel_initializer = "he_uniform", padding = "same")(p3)
    c4 = keras.layers.BatchNormalization()(c4)
    c4 = keras.layers.Activation("relu")(c4)
    c4 = keras.layers.Dropout(0.4)(c4)
    c4 = keras.layers.Conv2D(128, (3, 3), kernel_initializer = "he_uniform", padding = "same")(c4)
    c4 = keras.layers.BatchNormalization()(c4)
    c4 = keras.layers.Activation("relu")(c4)
    p4 = keras.layers.MaxPooling2D((2, 2))(c4)

    # Middle
    c5 = keras.layers.Conv2D(256, (3, 3), kernel_initializer = "he_uniform", padding = "same")(p4)
    c5 = keras.layers.BatchNormalization()(c5)
    c5 = keras.layers.Activation("relu")(c5)
    c5 = keras.layers.Dropout(0.5)(c5)
    c5 = keras.layers.Conv2D(256, (3, 3), kernel_initializer = "he_uniform", padding = "same")(c5)
    c5 = keras.layers.BatchNormalization()(c5)
    c5 = keras.layers.Activation("relu")(c5)

    # Expand (upscale) #1
    u6 = keras.layers.Conv2DTranspose(128, (3, 3), strides = (2, 2), padding = "same")(c5)
    u6 = keras.layers.concatenate([u6, c4])
    c6 = keras.layers.Conv2D(128, (3, 3), kernel_initializer = "he_uniform", padding = "same")(u6)
    c6 = keras.layers.BatchNormalization()(c6)
    c6 = keras.layers.Activation("relu")(c6)
    c6 = keras.layers.Dropout(0.5)(c6)
    c6 = keras.layers.Conv2D(128, (3, 3), kernel_initializer = "he_uniform", padding = "same")(c6)
    c6 = keras.layers.BatchNormalization()(c6)
    c6 = keras.layers.Activation("relu")(c6)

    # Expand (upscale) #2
    u7 = keras.layers.Conv2DTranspose(64, (3, 3), strides = (2, 2), padding = "same")(c6)
    u7 = keras.layers.concatenate([u7, c3])
    c7 = keras.layers.Conv2D(64, (3, 3), kernel_initializer = "he_uniform", padding = "same")(u7)
    c7 = keras.layers.BatchNormalization()(c7)
    c7 = keras.layers.Activation("relu")(c7)
    c7 = keras.layers.Dropout(0.5)(c7)
    c7 = keras.layers.Conv2D(64, (3, 3), kernel_initializer = "he_uniform", padding = "same")(c7)
    c7 = keras.layers.BatchNormalization()(c7)
    c7 = keras.layers.Activation("relu")(c7)

    # Expand (upscale) #3
    u8 = keras.layers.Conv2DTranspose(32, (3, 3), strides = (2, 2), padding = "same")(c7)
    u8 = keras.layers.concatenate([u8, c2])
    c8 = keras.layers.Conv2D(32, (3, 3), kernel_initializer = "he_uniform", padding = "same")(u8)
    c8 = keras.layers.BatchNormalization()(c8)
    c8 = keras.layers.Activation("relu")(c8)
    c8 = keras.layers.Dropout(0.5)(c8)
    c8 = keras.layers.Conv2D(32, (3, 3), kernel_initializer = "he_uniform", padding = "same")(c8)
    c8 = keras.layers.BatchNormalization()(c8)
    c8 = keras.layers.Activation("relu")(c8)

    # Expand (upscale) #4
    u9 = keras.layers.Conv2DTranspose(16, (3, 3), strides = (2, 2), padding = "same")(c8)
    u9 = keras.layers.concatenate([u9, c1])
    c9 = keras.layers.Conv2D(16, (3, 3), kernel_initializer = "he_uniform", padding = "same")(u9)
    c9 = keras.layers.BatchNormalization()(c9)
    c9 = keras.layers.Activation("relu")(c9)
    c9 = keras.layers.Dropout(0.5)(c9)
    c9 = keras.layers.Conv2D(16, (3, 3), kernel_initializer = "he_uniform", padding = "same")(c9)
    c9 = keras.layers.BatchNormalization()(c9)
    c9 = keras.layers.Activation("relu")(c9)

    output = keras.layers.Conv2D(1, (1, 1), activation = "sigmoid")(c9)
    model = keras.Model(inputs = [input_img], outputs = [output])
    return model

