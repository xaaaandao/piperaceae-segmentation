import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import pathlib
import pickle
import tensorflow as tf
import time

from image import image_segmented_background_transparency, image_segmented_background_white

def save_params(batch_size, channel, data_aug, epochs, folds, img_size, input_images, input_masks, learning_rate, loss_function, output, random_state, test_size, val_size):
    values = ["batch_size", "epochs", "learning_rate", "loss_function", "images", "masks", "byn_images", "byn_masks",
              "channel", "image_size", "fold", "test_size", "val_size", "random_state", "data_augmentation"]

    index = [batch_size, epochs, learning_rate, loss_function, input_images, input_masks,
             len(input_images), len(input_masks), channel, img_size, folds, test_size,
             val_size, random_state, data_aug]

    df = pd.DataFrame(index, values)
    filename = os.path.join(output, "params.csv")
    df.to_csv(filename, sep=";", quoting=2, na_rep="", encoding="utf-8", header=False)
    print("saving %s" % filename)


def save_folds(output, results):
    index = ["loss", "dice", "jaccard", "precision", "recall"]
    for evaluate in results:
        values_train = [evaluate["loss_train"], evaluate["dice_train"], evaluate["jaccard_train"],
                        evaluate["precision_train"], evaluate["recall_train"]]
        values_val = [evaluate["loss_val"], evaluate["dice_val"], evaluate["jaccard_val"], evaluate["precision_val"],
                      evaluate["recall_val"]]
        values_test = [evaluate["loss_test"], evaluate["dice_test"], evaluate["jaccard_test"],
                       evaluate["precision_test"], evaluate["recall_test"]]

        columns_and_values = {"metrics_train": values_train,
               "metrics_val": values_val,
               "metrics_test": values_test}

        output_dir = os.path.join(output, "fold-%d" % evaluate["fold"])
        os.makedirs(output_dir, exist_ok=True)
        df = pd.DataFrame(columns_and_values, index=index)
        filename = os.path.join(output_dir, "metrics.csv")
        df.to_csv(filename, sep=";", na_rep="", quoting=2)
        print("saving %s" % filename)


def get_mean(key, results):
    return str(np.mean([evaluate[key] for evaluate in results]))


def get_std(key, results):
    return str(np.std([evaluate[key] for evaluate in results]))


def get_mean_values(key, results):
    return [get_mean("loss_%s" % key, results),
            get_mean("dice_%s" % key, results),
            get_mean("jaccard_%s" % key, results),
            get_mean("precision_%s" % key, results),
            get_mean("recall_%s" % key, results)]


def get_std_values(key, results):
    return [get_std("loss_%s" % key, results),
            get_std("dice_%s" % key, results),
            get_std("jaccard_%s" % key, results),
            get_std("precision_%s" % key, results),
            get_std("recall_%s" % key, results)]


def save_mean(output, results):
    data = {"mean_train": get_mean_values("train", results),
            "std_train": get_std_values("train", results),
            "mean_val": get_mean_values("val", results),
            "std_val": get_std_values("val", results),
            "mean_test": get_mean_values("test", results),
            "std_test": get_std_values("test", results)}
    index = ["loss", "dice", "jaccard", "precision", "recall"]
    df = pd.DataFrame(data, index=index)
    filename = os.path.join(output, "means.csv")
    df.to_csv(filename, sep=";", quoting=2, na_rep="", encoding="utf-8")
    print("saving %s" % filename)


def get_min_value(key, results):
    min_value = min(results, key=lambda x: x[key])
    return {"fold": min_value["fold"], "value": min(results, key=lambda x: x[key])[key]}


def get_max_value(key, results):
    max_value = max(results, key=lambda x: x[key])
    return {"fold": max_value["fold"], "value": max(results, key=lambda x: x[key])[key]}


def save_best(output, results):
    data = {"loss_min_train": get_min_value("loss_train", results),
            "dice_max_train": get_max_value("dice_train", results),
            "jaccard_max_train": get_max_value("jaccard_train", results),
            "precision_max_train": get_max_value("precision_train", results),
            "recall_max_train": get_max_value("recall_train", results),
            "loss_min_val": get_min_value("loss_val", results),
            "dice_max_val": get_max_value("dice_val", results),
            "jaccard_max_val": get_max_value("jaccard_val", results),
            "precision_max_val": get_max_value("precision_val", results),
            "recall_max_val": get_max_value("recall_val", results),
            "loss_min_test": get_min_value("loss_test", results),
            "dice_max_test": get_max_value("dice_test", results),
            "jaccard_max_test": get_max_value("jaccard_test", results),
            "precision_max_test": get_max_value("precision_test", results),
            "recall_max_test": get_max_value("recall_test", results),
    }
    index = ["fold", "value"]
    df = pd.DataFrame(data, index=index)
    df = df.transpose()
    filename = os.path.join(output, "best.csv")
    df.to_csv(filename, sep=";", quoting=2, na_rep="", encoding="utf-8")
    print("saving %s" % filename)


def save_figs(figs, output):
    for f in figs:
        output_dir = os.path.join(output, "fold-%d" % f["fold"])
        for p in ["mask_unet", "w_pred_mask", "transparency"]:
            os.makedirs(os.path.join(output_dir, p), exist_ok=True)

        for i, idx in enumerate(f["index"]):
            image = f["x"][idx].reshape((1, f["img_size"], f["img_size"], f["channel"]))
            mask = save_mask(image, idx, f["masks"], f["model"], output_dir)
            image_original = image_segmented_background_transparency(idx, f["masks"], mask, output_dir, f["x"])
            image_segmented_background_white(f["channel"], image_original, f["img_size"], idx, f["masks"], output_dir)

def save_fit_history(f, output):
    output_dir = os.path.join(output, "fold-%d" % f["fold"])
    os.makedirs(output_dir, exist_ok=True)
    filename = os.path.join(output_dir, "fold-%d-fit.pckl" % f["fold"])
    try:
        with open(filename, "wb") as file:
            pickle.dump(f["fit"].history, file)
            file.close()
            print("saving %s" % filename)
    except Exception:
        raise SystemExit("error in create %s" % filename)

def save_fits(fits, output):
    for f in fits:
        save_fit_history(f, output)
        save_lossgraph(f, output)

def save_lossgraph(f, output):
    output_dir = os.path.join(output, "fold-%d" % f["fold"])
    os.makedirs(output_dir, exist_ok=True)
    filename = os.path.join(output_dir, "fold-%d-lossgraph.png" % f["fold"])
    figure, axis = plt.subplots(1, figsize=(10, 10))
    plt.ioff()
    axis.plot(f["fit"].history["loss"], label="Train")
    axis.plot(f["fit"].history["val_loss"], label="Validation")
    axis.plot(f["fit"].history["learning_rate"], label="Learning rate")
    figure.suptitle("Train, Validation and Learning Rate", fontsize=20, verticalalignment="center")
    axis.set_ylabel("Loss", fontsize=16)
    axis.set_xlabel("Epoch", fontsize=16)
    axis.legend()
    figure.savefig(filename)
    plt.cla()
    plt.clf()
    plt.close()
    print("saving %s" % filename)

def save_mask(image, index, images_names, model, output):
    mask = model.predict(image)
    mask = mask[0, :, :, :]
    filename = os.path.join(output, "mask_unet", "%s.bmp" % images_names[index].stem)
    tf.keras.preprocessing.image.save_img(filename, mask)
    print("saving %s" % filename)
    return mask

def save_results(output, results):
    for r in results:
        df = pd.DataFrame(r, index=list(r.keys()))
        df = df.transpose()
        output_dir = os.path.join(output, "fold-%d" % r["fold"])
        os.makedirs(output_dir, exist_ok=True)
        filename = os.path.join(output_dir, "results.csv")
        df.to_csv(filename, sep=";", quoting=2, na_rep="", encoding="utf-8")
        print("saving %s" % filename)

def save(batch_size, channel, data_aug, epochs, figs, fits, folds, img_size, input_images, input_masks, learning_rate, loss_function, output, random_state, results, test_size, val_size):
    save_best(output, results)
    save_figs(figs, output)
    save_fits(fits, output)
    save_folds(output, results)
    save_mean(output, results)
    save_params(batch_size, channel, data_aug, epochs, folds, img_size, input_images, input_masks, learning_rate, loss_function, output, random_state, test_size, val_size)
    # save_params(output)
    save_results(output, results)

