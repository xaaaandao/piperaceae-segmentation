import matplotlib.pyplot as plt
import numpy as np
import os
import pathlib
import pickle
import skimage
import tensorflow as tf

from PIL import Image

def create_folder(list_path):
    for path in list_path:
        pathlib.Path(path).mkdir(parents=True, exist_ok=True)


def save_figs(cfg, list_images_names, list_index, model, path, x):
    for p in ['mask_unet', 'w_pred_mask', 'transparency']:
        pathlib.Path(os.path.join(path, p)).mkdir(parents=True, exist_ok=True)

    for i, index in enumerate(list_index):
        filename = str(list_images_names[index].stem)
        image = x[index]/255
        image = image.reshape((1, cfg['image_size'], cfg['image_size'], cfg['channel']))
        mask = model.predict(image)
        mask = mask[0, :, :, :]
        new_filename = os.path.join(path, 'mask_unet', filename + '.bmp')
        print(new_filename, mask.shape)
        tf.keras.preprocessing.image.save_img(new_filename, mask)

        mask = tf.keras.preprocessing.image.array_to_img(mask).convert('L')
        image_original = tf.keras.preprocessing.image.load_img(list_images_names[index].resolve())
        image_original = image_original.convert('RGBA')
        new_filename = os.path.join(path, 'transparency', filename + '_transparente.png')
        image_original.putalpha(mask)
        image_original.save(new_filename)

        background = Image.new('RGBA', (cfg['image_size'], cfg['image_size']), "WHITE")
        img_w, img_h = image_original.size
        bg_w, bg_h = background.size
        offset = ((bg_w - img_w) // 2, (bg_h - img_h) // 2)
        background.paste(image_original, offset, image_original)
        new_filename = os.path.join(path, 'w_pred_mask', filename + '.png')
        background.save(new_filename)

def save_image_grayscale(cfg, list_images_names, image, index, model, path):
    image = image.reshape((1, cfg['image_size'], cfg['image_size'], cfg['channel']))
    x_pred = model.predict(image)[0, :, :, 0]
    pred_mask = np.uint8(x_pred >= 0.5)
    filename_pred_mask = list_images_names[index] + 'mask_unet.png'
    skimage.io.imsave(os.path.join(path, 'mask_unet', filename_pred_mask), skimage.img_as_ubyte(pred_mask * 255))

    img_segmented = image[0, :, :, 0] * pred_mask
    img_segmented[img_segmented == 0] = 1
    filename_image_pred_mask = list_images_names[index] + 'w_pred_mask.png'
    skimage.io.imsave(os.path.join(path, 'w_pred_mask', filename_image_pred_mask), skimage.img_as_ubyte(img_segmented))


def save_fit_history(fold, fit, path):
    filename = os.path.join(path, f'fold{fold}-fit.pckl')
    try:
        with open(filename, 'wb') as file:
            pickle.dump(fit.history, file)
            file.close()
            print(f'{filename} created')
    except Exception as e:
        raise SystemExit(f'error in create {filename}')


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