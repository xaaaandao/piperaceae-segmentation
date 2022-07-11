from image import get_all_images_and_masks
from test import cross_or_no


def main():
    X, Y = get_all_images_and_masks()
    all_cfg = [
            {"learning_rate": 0.001, "batch": 16, "epochs": 100, "val_size": 0.1},
            {"learning_rate": 0.05, "batch": 4, "epochs": 200, "val_size": 0.1},
            {"learning_rate": 0.05, "batch": 4, "epochs": 100, "val_size": 0.1},
            {"learning_rate": 0.05, "batch": 4, "epochs": 50, "val_size": 0.1},
        ]
    for cfg in all_cfg:
        cross_or_no(X, Y, cfg, fold=10)


if __name__ == '__main__':
    main()
