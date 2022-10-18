import matplotlib.pyplot as plt
import numpy as np
import os
import pathlib
import pickle
import skimage
import tensorflow as tf


def create_folder(list_path):
    for path in list_path:
        pathlib.Path(path).mkdir(parents=True, exist_ok=True)


def save_figs(cfg, list_images_names, list_index, model, path, x):
    for p in ['mask_unet', 'w_pred_mask']:
        pathlib.Path(os.path.join(path, p)).mkdir(parents=True, exist_ok=True)

    for i, index in enumerate(list_index):
        filename = list_images_names[index]
        # image = x[index] / 255
        image = x[index].reshape((1, cfg['image_size'], cfg['image_size'], cfg['channel']))
        mask = model.predict(image)
        mask = mask[0, :, :, :]
        new_filename = os.path.join(path, 'mask_unet', filename + '.bmp')
        print(new_filename, mask.shape)
        tf.keras.preprocessing.image.save_img(new_filename, mask)

        mask = np.uint8(mask >= 0.5)
        image_segmented = image * mask
        image_segmented[image_segmented == 0] = 1
        image_segmented = image_segmented[0, :, :, :]
        image_segmented = tf.keras.preprocessing.image.array_to_img(image_segmented)
        new_filename = os.path.join(path, 'w_pred_mask', filename + '.png')
        tf.keras.preprocessing.image.save_img(new_filename, image_segmented)
    #     if cfg['channel'] == 3:
    #         save_image_rgb(cfg, list_images_names, x[index], index, model, path)
    #     else:
    #         save_image_rgb(cfg, list_images_names, x[index], index, model, path)



    # print(x[index].shape)
    # image_segmented = image * pred_mask[0, :, :, :]
    # image_segmented[image_segmented == 0] = 1
    # filename_image_pred_mask = list_images_names[index] + 'w_pred_mask.png'
    # skimage.io.imsave(os.path.join(path, 'w_pred_mask', filename_image_pred_mask), skimage.img_as_ubyte(image_segmented))


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