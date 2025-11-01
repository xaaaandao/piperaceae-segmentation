import os

import tensorflow as tf
from PIL import Image

def set_color_mode_background(background, channel):
    return background.convert("RGB") if channel == 3 else background.convert("L")


def image_segmented_background_transparency(index, list_images_names, mask, path, x):
    mask = tf.keras.preprocessing.image.array_to_img(mask).convert("L")
    image = tf.keras.preprocessing.image.array_to_img(x[index])
    image_original = image.convert("RGBA")
    new_filename = os.path.join(path, "transparency", str(list_images_names[index].stem) + "_transparente.png")
    image_original.putalpha(mask)
    image_original.save(new_filename)
    return image_original


def image_segmented_background_white(channel, image_original, img_size, index, list_images_names, path):
    background = Image.new("RGBA", (img_size, img_size), "WHITE")
    image_width, image_height = image_original.size
    background_width, background_height = background.size
    offset = ((background_width - image_width) // 2, (background_height - image_height) // 2)
    background.paste(image_original, offset, image_original)
    filename = os.path.join(path, "w_pred_mask", str(list_images_names[index].stem) + ".jpeg")
    background = set_color_mode_background(background, channel)
    background.save(filename, format="jpeg")



