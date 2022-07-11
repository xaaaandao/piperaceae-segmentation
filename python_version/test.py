import os

import keras.backend
import sklearn

from file import create_file_result_final, create_dir_if_not_exists, path_out
from image import load_all_data, get_xy_val
from unet import train_test


#     for (train_index, test_index) in itertools.islice(indices.split(x, y), 9, None):
#         i = 9
def cross_validation(x, y, cfg, fold):
    indices = sklearn.model_selection.StratifiedKFold(n_splits=fold, shuffle=True, random_state=1234)
    sum_metrics = {"iou_train": 0, "dice_train": 0, "iou_val": 0, "dice_val": 0, "iou_test": 0, "dice_test": 0}
    for i, (train_index, test_index) in enumerate(indices.split(x, y)):
        path_out_index_cv = os.path.join(path_out, f"cv-{i}")
        create_dir_if_not_exists(path_out_index_cv)

        x_train, y_train = x[train_index], x[train_index]
        x_test, y_test = x[test_index], x[test_index]
        x_train, x_val, y_train, y_val = get_xy_val(x_train, y_train, cfg)
        x_train, y_train, x_val, y_val, x_test, y_test = load_all_data(x_train, x_val, x_test, path_out_index_cv)
        train_test(x_train, y_train, x_val, y_val, x_test, y_test, path_out_index_cv, cfg, sum_metrics=sum_metrics)

    create_file_result_final(path_out, sum_metrics, fold)


def no_cross(x, y, cfg):
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=cfg["test_size"],
                                                                                random_state=1234)
    x_train, x_val, y_train, y_val = get_xy_val(x_train, y_train, cfg)
    x_train, y_train, x_val, y_val, x_test, y_test = load_all_data(x_train, x_val, x_test, path_out)
    train_test(x_train, y_train, x_val, y_val, x_test, y_test, path_out, cfg)


def cross_or_no(x, y, cfg, fold=None):
    keras.backend.clear_session()
    create_dir_if_not_exists(path_out)
    if fold:
        cross_validation(x, y, cfg, fold)
    else:
        no_cross(x, y, cfg)
    keras.backend.clear_session()
