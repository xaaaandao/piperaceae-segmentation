import os

import click
import csv
import pandas as pd
import pathlib

ROUND_VALUE = 4


def round_mean(value):
    return '=ROUND(' + str(value) + '; ' + str(ROUND_VALUE) + ')'


def plus_minus_std(value):
    return '="±"&ROUND(' + str(value) + '; ' + str(ROUND_VALUE) + ')'


def get_mean(sheet, metric):
    if metric not in ['train', 'val', 'test']:
        raise ValueError('metric not valid')

    mean_metric = 'mean_' + metric
    std_metric = 'std_' + metric
    return [
        round_mean(sheet.loc['loss'][mean_metric]), plus_minus_std(sheet.loc['loss'][std_metric]),
        round_mean(sheet.loc['dice'][mean_metric]), plus_minus_std(sheet.loc['dice'][std_metric]),
        round_mean(sheet.loc['jaccard'][mean_metric]), plus_minus_std(sheet.loc['jaccard'][std_metric]),
        round_mean(sheet.loc['precision'][mean_metric]), plus_minus_std(sheet.loc['precision'][std_metric]),
        round_mean(sheet.loc['recall'][mean_metric]), plus_minus_std(sheet.loc['recall'][std_metric]),
    ]


def insert_values(image_size, mean_train, mean_val, mean_test, sheet):
    sheet['mean_train_' + str(image_size)] = mean_train
    sheet['mean_val_' + str(image_size)] = mean_val
    sheet['mean_test_' + str(image_size)] = mean_test


def save_xlsx_csv(filename, sheet):
    sheet.to_excel(filename + '.xlsx', na_rep='', engine='xlsxwriter')
    print(f'save {filename}.xlsx')
    sheet.to_csv(filename + '.csv', sep=';', na_rep='', quoting=csv.QUOTE_ALL, index=None)
    print(f'save {filename}.csv')


def data_is_grayscale(color_mode):
    return color_mode == str(1)


def insert_sheet(color_mode, image_size, mean_train, mean_val, mean_test, sheet_grayscale, sheet_rgb):
    return insert_values(image_size, mean_train, mean_val, mean_test, sheet_grayscale) if data_is_grayscale(color_mode) else insert_values(image_size, mean_train, mean_val, mean_test, sheet_rgb)


@click.command()
@click.option(
    '--path',
    '-p',
    type=str,
    required=True
)
def main(path):
    if not os.path.exists(path):
        raise ValueError(f'{path} not exists')

    if not os.path.isdir(path):
        raise ValueError(f'{path} not is dir')

    list_files = [file for file in pathlib.Path(path).rglob('mean.csv') if file.is_file()]

    if len(list_files) == 0:
        raise FileNotFoundError(f'files not found in dir {path}')

    index = ['loss', 'dice', 'jaccard', 'precision', 'recall']
    index = [i + '_' + mean_std for i in index for mean_std in ['mean', 'std']]
    sheet_rgb = pd.DataFrame(index)
    sheet_grayscale = pd.DataFrame(index)

    for file in sorted(list_files):
        sheet_cfg = pd.read_csv(str(file).replace('mean.csv', 'cfg.csv'), header=None, sep=';', index_col=0)
        color_mode = sheet_cfg.loc['channel'][1]
        image_size = sheet_cfg.loc['image_size'][1]

        sheet_mean = pd.read_csv(file, sep=';', index_col=0)
        mean_train = get_mean(sheet_mean, 'train')
        mean_val = get_mean(sheet_mean, 'val')
        mean_test = get_mean(sheet_mean, 'test')
        insert_sheet(color_mode, image_size, mean_train, mean_val, mean_test, sheet_grayscale, sheet_rgb)

        save_xlsx_csv('mean_grayscale', sheet_grayscale)
        save_xlsx_csv('mean_rgb', sheet_rgb)


if __name__ == '__main__':
    main()