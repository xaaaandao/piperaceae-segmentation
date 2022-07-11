from image import get_all_images_and_masks
from test import cross_or_no


def main():
    X, Y = get_all_images_and_masks()
    cfg = {"learning_rate": 0.001, "batch": 16, "epochs": 4, "val_size": 0.1, "test_size": 0.1}
    cross_or_no(X, Y, cfg)


if __name__ == '__main__':
    main()
