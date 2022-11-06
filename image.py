import os

import tensorflow as tf
from PIL import Image
from matplotlib import pyplot as plt

from files import create_dir


def image_is_rgb(cfg):
    return cfg['channel'] == 3


def set_color_mode_background(background, cfg):
    return background.convert('RGB') if image_is_rgb(cfg) else background.convert('L')


def save_figs(cfg, list_images_names, list_index, model, path, x):
    create_dir([os.path.join(path, 'mask_unet'), os.path.join(path, 'w_pred_mask'), os.path.join(path, 'transparency')])

    for i, index in enumerate(list_index):
        image = x[index].reshape((1, cfg['image_size'], cfg['image_size'], cfg['channel']))
        mask = save_mask(image, index, list_images_names, model, path)
        image_original = image_segmented_background_transparency(index, list_images_names, mask, path, x)
        image_segmented_background_white(cfg, image_original, index, list_images_names, path)


def save_mask(image, index, list_images_names, model, path):
    mask = model.predict(image)
    mask = mask[0, :, :, :]
    filename = os.path.join(path, 'mask_unet', str(list_images_names[index].stem) + '.bmp')
    tf.keras.preprocessing.image.save_img(filename, mask)
    return mask


def image_segmented_background_transparency(index, list_images_names, mask, path, x):
    mask = tf.keras.preprocessing.image.array_to_img(mask).convert('L')
    image = tf.keras.preprocessing.image.array_to_img(x[index])
    image_original = image.convert('RGBA')
    new_filename = os.path.join(path, 'transparency', str(list_images_names[index].stem) + '_transparente.png')
    image_original.putalpha(mask)
    image_original.save(new_filename)
    return image_original


def image_segmented_background_white(cfg, image_original, index, list_images_names, path):
    background = Image.new('RGBA', (cfg['image_size'], cfg['image_size']), 'WHITE')
    image_width, image_height = image_original.size
    background_width, background_height = background.size
    offset = ((background_width - image_width) // 2, (background_height - image_height) // 2)
    background.paste(image_original, offset, image_original)
    filename = os.path.join(path, 'w_pred_mask', str(list_images_names[index].stem) + '.jpeg')
    background = set_color_mode_background(background, cfg)
    background.save(filename, format='jpeg')


def save_lossgraph(fold, model, path):
    filename = os.path.join(path, f'fold{fold}-lossgraph.png')
    figure, axis = plt.subplots(1, figsize=(10, 10))
    plt.ioff()
    axis.plot(model.history['loss'], label='Train')
    axis.plot(model.history['val_loss'], label='Validation')
    axis.plot(model.history['lr'], label='Learning rate')
    figure.suptitle('Train, Validation and Learning Rate', fontsize=20, verticalalignment='center')
    axis.set_ylabel('Loss', fontsize=16)
    axis.set_xlabel('Epoch', fontsize=16)
    axis.legend()
    figure.savefig(filename)
    plt.cla()
    plt.clf()
    plt.close()
    print(f'{filename} created')
