import datetime
import time
import matplotlib.pyplot
import numpy
import os
import pathlib
import pickle

import skimage


def create_folder(list_path):
    for path in list_path:
        pathlib.Path(path).mkdir(parents=True, exist_ok=True)


def save_figs(cfg, list_images_names, list_index, model, path, x):
    for index in list_index:
        image = x[index, :, :, :]
        x_pred = model.predict(image.reshape((1, cfg["image_size"], cfg["image_size"], cfg["channel"])))
        pred_mask = numpy.uint8(x_pred >= 0.5)
        filename_pred_mask = list_images_names[index] + "mask_unet.png"
        skimage.io.imsave(os.path.join(path, filename_pred_mask), skimage.img_as_ubyte(pred_mask[0, :, :, 0] * 255))

        if cfg["channel"] == 1:
            image_pred_mask = image_grayscale(image, pred_mask)
        elif cfg["channel"] == 3:
            image_pred_mask = image_rgb(image, pred_mask)
        filename_image_pred_mask = list_images_names[index] + "w_pred_mask.png"
        skimage.io.imsave(os.path.join(path, filename_image_pred_mask), skimage.img_as_ubyte(image_pred_mask))


def image_rgb(image, pred_mask):
    image_pred_mask = image * pred_mask
    image_pred_mask = image_pred_mask[0, :, :, :]  # reduz a dimensionalidade
    image_pred_mask[image_pred_mask == 0] = 1
    return image_pred_mask


def image_grayscale(image, pred_mask):
    image_pred_mask = numpy.uint8(image) * pred_mask
    image_pred_mask = image_pred_mask[0, :, :, :]
    image_pred_mask[image_pred_mask == 0] = 1
    image_pred_mask = image_pred_mask * 255
    return image_pred_mask


def save_fit_history(fold, fit, path):
    filename = os.path.join(path, f"fold{fold}-fit.pckl")
    try:
        with open(filename, "wb") as file:
            pickle.dump(fit.history, file)
            file.close()
            print(f"{filename} created")
    except Exception as e:
        raise SystemExit(f"error in create {filename}")


def save_lossgraph(fold, model, path):
    filename = os.path.join(path, f"fold{fold}-lossgraph.png")
    figure, axis = matplotlib.pyplot.subplots(1, figsize=(10, 10))
    matplotlib.pyplot.ioff()
    axis.plot(model.history["loss"], label="Train")
    axis.plot(model.history["val_loss"], label="Validation")
    axis.plot(model.history["lr"], label="Learning rate")
    figure.suptitle("Train, Validation and Learning Rate", fontsize=20, verticalalignment="center")
    axis.set_ylabel("Loss", fontsize=16)
    axis.set_xlabel("Epoch", fontsize=16)
    axis.legend()
    figure.savefig(filename)
    matplotlib.pyplot.cla()
    matplotlib.pyplot.clf()
    matplotlib.pyplot.close()
    print(f"{filename} created")