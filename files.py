import datetime
import sys

import numpy
import os
import pathlib
import pickle
import skimage.io

from image import pred_mask, apply_mask, convert_background_color
from plots import plot_lossgraph


def load_mask(filename):
    return numpy.float32(skimage.io.imread(filename) / 255)


def load_image(filename):
    return skimage.img_as_float32(skimage.io.imread(filename))


def get_format_time_to_filename():
    return datetime.datetime.now().strftime('%d-%m-%Y-%H-%M-%S')


def get_path_outfile_each_fold(fold, path):
    return os.path.join(path, fold)


def get_path_mean(cfg):
    return os.path.join(cfg["path_out"], get_format_time_to_filename())


def create_outfile_each_fold(fold, metrics, path):
    filename = os.path.join(path, "out.txt")
    try:
        with open(filename, "w") as file:
            file.write(f"fold: {fold}\n")
            file.write(f"dice_test: {getattr(metrics, 'dice_test')}, iou_test: {getattr(metrics, 'iou_test')}\n")
            file.write(f"dice_train: {getattr(metrics, 'dice_train')}, iou_train: {getattr(metrics, 'iou_train')}\n")
            file.write(f"dice_val: {getattr(metrics, 'dice_val')}, iou_val: {getattr(metrics, 'iou_val')}\n")
            file.close()
    except Exception as e:
        raise SystemError(f"problems in file {e}")


def create_outfile_mean(cfg, list_result, path):
    mean_dice_test = numpy.mean(numpy.array([getattr(l, "dice_test") for l in list_result]))
    std_dice_test = numpy.std(numpy.array([getattr(l, "dice_test") for l in list_result]))
    mean_dice_train = numpy.mean(numpy.array([getattr(l, "dice_train") for l in list_result]))
    std_dice_train = numpy.std(numpy.array([getattr(l, "dice_train") for l in list_result]))
    mean_dice_val = numpy.mean(numpy.array([getattr(l, "dice_val") for l in list_result]))
    std_dice_val = numpy.std(numpy.array([getattr(l, "dice_val") for l in list_result]))
    mean_iou_test = numpy.mean(numpy.array([getattr(l, "iou_test") for l in list_result]))
    std_iou_test = numpy.std(numpy.array([getattr(l, "iou_test") for l in list_result]))
    mean_iou_train = numpy.mean(numpy.array([getattr(l, "iou_train") for l in list_result]))
    std_iou_train = numpy.std(numpy.array([getattr(l, "iou_train") for l in list_result]))
    mean_iou_val = numpy.mean(numpy.array([getattr(l, "iou_val") for l in list_result]))
    std_iou_val = numpy.std(numpy.array([getattr(l, "iou_val") for l in list_result]))
    filename = os.path.join(path, "out.txt")
    try:
        with open(filename, "w") as file:
            file.write(f"mean_dice_test: {mean_dice_test}, std_dice_test: {std_dice_test}\n")
            file.write(f"mean_dice_train: {mean_dice_train}, std_dice_train: {std_dice_train}\n")
            file.write(f"mean_dice_val: {mean_dice_val}, std_dice_val: {std_dice_val}\n")
            file.write(f"mean_iou_test: {mean_iou_test}, std_iou_test: {std_iou_test}\n")
            file.write(f"mean_iou_train: {mean_iou_train}, std_iou_train: {std_iou_train}\n")
            file.write(f"mean_iou_val: {mean_iou_val}, std_iou_val: {std_iou_val}\n")
            file.close()
    except Exception as e:
        raise SystemError(f"problems in file {e}")


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
    figure = plot_lossgraph(model)
    print(f"{filename} created")
    figure.savefig(filename)


def save_image(filename, image, path):
    filename = os.path.join(path, filename)
    skimage.io.imsave(filename, skimage.img_as_ubyte(image))


def predict_and_save(cfg, model, path, x, y):
    for i, image in enumerate(x):
        p = os.path.join(path, str(i))
        pathlib.Path(p).mkdir(parents=True, exist_ok=True)

        file = getattr(image["image"], "file")
        image_original = getattr(image["image"], "image")
        mask_original = getattr(image["mask"], "image")
        image_original = image_original.reshape((1, cfg["image_size"], cfg["image_size"], 1))
        mask_original = mask_original.reshape((1, cfg["image_size"], cfg["image_size"], 1))
        mask_original = numpy.uint8(mask_original[0, :, :, 0] > 0.5)

        mask_unet = pred_mask(image_original, model)
        image_with_mask_original = apply_mask(image_original, mask_original)
        image_with_mask_unet = apply_mask(image_original, numpy.uint8(mask_unet > 0.5))
        convert_background_color(image_with_mask_original)
        convert_background_color(image_with_mask_unet)
        save_image(f"{file.stem}+original.png", image_original[0, :, :, 0], p)
        save_image(f"{file.stem}+mask_original.png", mask_original*255, p)
        save_image(f"{file.stem}+mask_unet.png", mask_unet, p)
        save_image(f"{file.stem}+img+mask_original.png", image_with_mask_original, p)
        save_image(f"{file.stem}+img+mask_unet.png", skimage.img_as_ubyte(image_with_mask_unet), p)
        sys.exit()


def save_figs(cfg, model, path, x_test, x_train, x_val, y_test, y_train, y_val):
    path_test = os.path.join(path, "test")
    pathlib.Path(path_test).mkdir(parents=True, exist_ok=True)
    path_train = os.path.join(path, "train")
    pathlib.Path(path_train).mkdir(parents=True, exist_ok=True)
    path_val = os.path.join(path, "val")
    pathlib.Path(path_val).mkdir(parents=True, exist_ok=True)
    predict_and_save(cfg, model, path_test, x_test, y_test)
    # predict_and_save(cfg, model, path_train, x_train, y_train)
    # predict_and_save(cfg, model, path_val, x_val, y_val)
