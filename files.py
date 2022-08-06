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


def save_figs(list_images_names, model, path, x):
    x_pred = model.predict(x)
    for idx in range(x_pred.shape[0]):
        print(f"save img {list_images_names[idx]}")
        pred_mask = numpy.uint8(x_pred[idx] >= 0.5)
        filename_pred_mask = list_images_names[idx] + "mask_unet.png"
        skimage.io.imsave(os.path.join(path, filename_pred_mask), skimage.img_as_ubyte(pred_mask * 255))

        image = x[idx]
        image_pred_mask = image * pred_mask
        image_pred_mask[image_pred_mask == 0] = 1
        filename_image_pred_mask = list_images_names[idx] + "w_pred_mask.png"
        skimage.io.imsave(os.path.join(path, filename_image_pred_mask), skimage.img_as_ubyte(image_pred_mask))


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