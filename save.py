import csv
import numpy as np
import os
import pandas as pd
import pathlib
import time

from files import create_dir


def save_cfg(cfg, filename, images_folder, list_images, list_labels, masks_folder):
    values = ['batch_size', 'epochs', 'learning_rate', 'loss_function', 'images', 'masks', 'len_images', 'len_masks',
              'channel', 'image_size', 'fold', 'test_size', 'val_size', 'random_state', 'path_dataset', 'path_out',
              'data_augmentation', 'filename_script']

    index = [cfg['batch_size'], cfg['epochs'], cfg['learning_rate'], cfg['loss_function'], images_folder, masks_folder,
             len(list_images), len(list_labels), cfg['channel'], cfg['image_size'], cfg['fold'], cfg['test_size'],
             cfg['val_size'], cfg['random_state'], cfg['path_dataset'], cfg['path_out'], cfg['data_augmentation'],
             str(filename)]

    return pd.DataFrame(index, values)


def save_fold(list_evaluate, path):
    index = ['loss', 'dice', 'jaccard', 'precision', 'recall']
    for evaluate in list_evaluate:
        values_train = [evaluate['loss_train'], evaluate['dice_train'], evaluate['jaccard_train'],
                        evaluate['precision_train'], evaluate['recall_train']]
        values_val = [evaluate['loss_val'], evaluate['dice_val'], evaluate['jaccard_val'], evaluate['precision_val'],
                      evaluate['recall_val']]
        values_test = [evaluate['loss_test'], evaluate['dice_test'], evaluate['jaccard_test'],
                       evaluate['precision_test'], evaluate['recall_test']]

        columns_and_values = {'metrics_train': values_train,
                              'metrics_val': values_val,
                              'metrics_test': values_test}

        path_to_csv = os.path.join(path, str(evaluate['fold']), 'csv')
        pathlib.Path(path_to_csv).mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame(columns_and_values, index=index)
        df.to_csv(os.path.join(path_to_csv, 'metrics.csv'), sep=';', na_rep='', quoting=csv.QUOTE_ALL)
        df.to_excel(os.path.join(path, str(evaluate['fold']), 'metrics.xlsx'), na_rep='', engine='xlsxwriter')


def get_mean(key, list_evaluate):
    return str(np.mean(list([evaluate[key] for evaluate in list_evaluate])))


def get_std(key, list_evaluate):
    return str(np.std(list([evaluate[key] for evaluate in list_evaluate])))


def get_mean_values(key, list_evaluate):
    return [get_mean(f'loss_{key}', list_evaluate),
            get_mean(f'dice_{key}', list_evaluate),
            get_mean(f'jaccard_{key}', list_evaluate),
            get_mean(f'precision_{key}', list_evaluate),
            get_mean(f'recall_{key}', list_evaluate)]


def get_std_values(key, list_evaluate):
    return [get_std(f'loss_{key}', list_evaluate),
            get_std(f'dice_{key}', list_evaluate),
            get_std(f'jaccard_{key}', list_evaluate),
            get_std(f'precision_{key}', list_evaluate),
            get_std(f'recall_{key}', list_evaluate)]


def save_mean_time(list_time):
    mean_time = np.mean(list_time)
    mean_time_seconds = time.strftime('%H:%M:%S', time.gmtime(mean_time))
    std_time = np.std(list_time)

    index = ['mean_time', 'mean_time_sec', 'std_time']
    values = [mean_time, mean_time_seconds, std_time]
    return pd.DataFrame(values, index=index)


def save_mean(list_evaluate):
    columns_and_values = {'mean_train': get_mean_values('train', list_evaluate),
                          'std_train': get_std_values('train', list_evaluate),
                          'mean_val': get_mean_values('val', list_evaluate),
                          'std_val': get_std_values('val', list_evaluate),
                          'mean_test': get_mean_values('test', list_evaluate),
                          'std_test': get_std_values('test', list_evaluate)}
    index = ['loss', 'dice', 'jaccard', 'precision', 'recall']
    return pd.DataFrame(columns_and_values, index=index)


def get_min_value(key, list_evaluate):
    min_value = min(list_evaluate, key=lambda x: x[key])
    return {'fold': min_value['fold'], 'value': min(list_evaluate, key=lambda x: x[key])[key]}


def get_max_value(key, list_evaluate):
    max_value = max(list_evaluate, key=lambda x: x[key])
    return {'fold': max_value['fold'], 'value': max(list_evaluate, key=lambda x: x[key])[key]}


def save_best(list_evaluate):
    index = ['fold', 'value']
    columns_and_values = {'loss_min_train': get_min_value('loss_train', list_evaluate),
                          'dice_max_train': get_max_value('dice_train', list_evaluate),
                          'jaccard_max_train': get_max_value('jaccard_train', list_evaluate),
                          'precision_max_train': get_max_value('precision_train', list_evaluate),
                          'recall_max_train': get_max_value('recall_train', list_evaluate),
                          'loss_min_val': get_min_value('loss_val', list_evaluate),
                          'dice_max_val': get_max_value('dice_val', list_evaluate),
                          'jaccard_max_val': get_max_value('jaccard_val', list_evaluate),
                          'precision_max_val': get_max_value('precision_val', list_evaluate),
                          'recall_max_val': get_max_value('recall_val', list_evaluate),
                          'loss_min_test': get_min_value('loss_test', list_evaluate),
                          'dice_max_test': get_max_value('dice_test', list_evaluate),
                          'jaccard_max_test': get_max_value('jaccard_test', list_evaluate),
                          'precision_max_test': get_max_value('precision_test', list_evaluate),
                          'recall_max_test': get_max_value('recall_test', list_evaluate),
                          }
    df = pd.DataFrame(columns_and_values, index=index)

    return df.transpose()


def save_xlsx(best, cfg, mean, mean_time, path):
    writer = pd.ExcelWriter(os.path.join(path, f'result.xlsx'), engine='xlsxwriter')
    best.to_excel(writer, sheet_name='best', na_rep='')
    cfg.to_excel(writer, sheet_name='cfg', na_rep='', header=False)
    mean.to_excel(writer, sheet_name='mean', na_rep='')
    mean_time.to_excel(writer, sheet_name='mean_time', na_rep='', header=False)
    writer.save()


def save_csv(best, cfg, mean, mean_time, path):
    path = os.path.join(path, 'csv')
    create_dir([path])
    best.to_csv(os.path.join(path, 'best.csv'), sep=';', na_rep='', quoting=csv.QUOTE_ALL)
    cfg.to_csv(os.path.join(path, 'cfg.csv'), sep=';', na_rep='', quoting=csv.QUOTE_ALL, header=False)
    mean.to_csv(os.path.join(path, 'mean.csv'), sep=';', na_rep='', quoting=csv.QUOTE_ALL)
    mean_time.to_csv(os.path.join(path, 'mean_time.csv'), sep=';', na_rep='', quoting=csv.QUOTE_ALL, header=False)


def save(cfg, filename, images_folder, list_evaluate, list_images, list_labels, list_time, masks_folder, path):
    best = save_best(list_evaluate)
    cfg = save_cfg(cfg, filename, images_folder, list_images, list_labels, masks_folder)
    mean = save_mean(list_evaluate)
    mean_time = save_mean_time(list_time)
    save_fold(list_evaluate, path)
    save_xlsx(best, cfg, mean, mean_time, path)
    save_csv(best, cfg, mean, mean_time, path)