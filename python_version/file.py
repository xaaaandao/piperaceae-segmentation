import os
import pathlib
import pickle
import re

from metrics import mean_all_metrics

path_images = "../images"
path_masks = "../masks"
path_out = "result"


def create_dir_if_not_exists(dir):
    if not os.path.isdir(dir):
        os.makedirs(dir)


def get_genus(filename):
    list_genus = ["manekia", "ottonia", "peperomia", "piper", "pothomorphe"]
    return next((g for g in list_genus if g in filename), ValueError(f"{filename} is not formatted"))


def get_path_mask(filename):
    filename = re.sub(r".[a-z]*$", "", filename)
    list_mask = list([str(file.resolve()) for file in pathlib.Path(path_masks).rglob("*")])
    return next((m for m in list_mask if filename in m), ValueError("a"))


def create_file_result_cv(x_test, x_train, x_val, cfg, metrics, index_cv, path, unet_filename, elapsed_time):
    try:
        print(f"{os.path.join(path, f'cv-{index_cv}-result.txt')} created")
        with open(os.path.join(path, f"cv-{index_cv}-result.txt"), "w") as file:
            file.write(f"unet filename={unet_filename}\n")
            file.write(f"index cv={index_cv}\n")
            file.write(f"elapsed time={elapsed_time}\n")
            file.write(f"X_train: {x_train.shape}, X_val: {x_val.shape}, X_test: {x_test.shape}\n")
            for key, value in cfg.items():
                file.write(f"{key}: {value}\n")
            file.write(f"============================================\n")
            for key, value in metrics.items():
                file.write(f"{key}: {value}\n")
            file.close()
    except FileExistsError:
        raise SystemError(f"fail in create outfile {os.path.join(path, f'cv-{str(index_cv)}.txt')}")


def create_file_result_final(path, metrics, fold):
    mean_all_metrics(metrics, fold)
    try:
        print(f"{os.path.join(path, f'mean-fold{fold}')}.txt created")
        with open(f"{os.path.join(path, f'mean-fold{fold}')}.txt", "w") as file:
            for key, value in metrics.items():
                file.write(f"Mean {key}: {value}\n")
            file.close()
    except FileExistsError:
        raise SystemError(f"fail in create outfile {os.path.join(path, 'mean')}.txt")


def create_images_used(x, path, type):
    try:
        print(f"{os.path.join(path, f'{type}-images-used.csv')} created")
        with open(os.path.join(path, f"{type}-images-used.csv"), "w") as file:
            file.write("index;path_image;path_mask\n")
            for image in x:
                file.write(f"\"{list(image.values())[0]}\";\"{list(image.values())[1]}\"\n")
            file.close()
    except FileExistsError:
        raise SystemExit(f"error in create file")


def save_fit_history(fit, index_cv, path):
    try:
        with open(os.path.join(path, f"cv{index_cv}-fit.pckl"), "wb") as file:
            pickle.dump(fit.history, file)
            file.close()
    except FileExistsError:
        raise SystemExit(f"error in create cv{index_cv}-fit.pckl")


def get_path_all_images():
    return list(
        [{"path_image": str(file.resolve()), "path_mask": get_path_mask(file.name), "label": get_genus(str(file.name))}
         for file in pathlib.Path(path_images).rglob("*")])
