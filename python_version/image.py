import os
import matplotlib
import numpy
import pathlib
import skimage
import skimage.io
import skimage.morphology
import sklearn
import threading

from file import create_dir_if_not_exists, get_path_all_images, create_images_used

image_size = 400


def load_mask(filename):
    return numpy.float32(skimage.io.imread(filename) / 255)


def load_all_masks(path):
    return [load_mask(filename) for filename in sorted(pathlib.Path(path).rglob("*"))]


def load_image(filename):
    return skimage.img_as_float32(skimage.io.imread(filename))


def load_all_images(path):
    return [load_image(filename) for filename in sorted(pathlib.Path(path).rglob("*"))]


def width_height_are_equal_image_size(image):
    return image.shape[0] == image_size and image.shape[1] == image_size


def validate_data(path):
    return all(not width_height_are_equal_image_size(skimage.io.imread(str(filename))) for filename in
               sorted(pathlib.Path(path).rglob("*")))


def format_data(x, y):
    return x.reshape(x.shape[0], image_size, image_size, 1), y.reshape(y.shape[0], image_size, image_size, 1)


def plot_loss_graph(model, index_cv, path):
    fig, ax = matplotlib.pyplot.subplots(1, figsize=(10, 10))
    ax.plot(model.history["loss"], label="Train", linewidth=2)
    ax.plot(model.history["val_loss"], label="Validation", linewidth=2)
    ax.plot(model.history["lr"], label="Learning rate", linewidth=2)
    fig.suptitle("Train, Validation and Learning Rate", fontsize=20, verticalalignment="center")
    ax.set_ylabel("Loss", fontsize=16)
    ax.set_xlabel("Epoch", fontsize=16)
    ax.legend()
    print(f"{os.path.join(path, f'cv-{index_cv}-lossgraph.png')} created")
    fig.savefig(os.path.join(path, f"cv-{index_cv}-lossgraph.png"))


def process_pred_mask(pred_mask):
    open_pred_mask = skimage.morphology.erosion(pred_mask, skimage.morphology.square(5))
    open_pred_mask = skimage.morphology.dilation(open_pred_mask, skimage.morphology.square(5))
    return skimage.morphology.dilation(open_pred_mask, skimage.morphology.square(5))


def save_image_mask_predmask(filename, image, mask, pred_mask, post_pred_mask, image_original_mask, image_pred_mask):
    figure = matplotlib.pyplot.figure(figsize=(15, 10))
    figure.add_subplot(2, 3, 1).set_title("Original image", fontdict={"fontsize": 18})
    matplotlib.pyplot.imshow(skimage.img_as_ubyte(image), cmap="gray")
    figure.add_subplot(2, 3, 2).set_title("Original mask", fontdict={"fontsize": 18})
    matplotlib.pyplot.imshow(skimage.img_as_ubyte(mask), cmap="gray")
    figure.add_subplot(2, 3, 3).set_title("Predicted mask", fontdict={"fontsize": 18})
    matplotlib.pyplot.imshow(pred_mask, cmap="gray")
    figure.add_subplot(2, 3, 4).set_title("Preprocessed mask", fontdict={"fontsize": 18})
    matplotlib.pyplot.imshow(post_pred_mask, cmap="gray")
    figure.add_subplot(2, 3, 5).set_title("Image original mask", fontdict={"fontsize": 18})
    matplotlib.pyplot.imshow(image_original_mask, cmap="gray")
    figure.add_subplot(2, 3, 6).set_title("Image pred mask", fontdict={"fontsize": 18})
    matplotlib.pyplot.imshow(image_pred_mask, cmap="gray")
    print(f"{filename} created")
    figure.savefig(filename)


def apply_mask_in_image(image, mask):
    return image[0, :, :, 0] * mask


def change_color_background(image):
    image[image == 0] = 1


def save_images(x, y, data_type, model, path):
    for idx in range(0, x.shape[0]):
        test_img, test_mask = get_image_by_idx(idx, x, y)
        pred_mask = model.predict(test_img)[0, :, :, 0]
        image_pred_mask = pred_mask
        pred_mask = numpy.uint8(pred_mask > 0.5)
        post_pred_mask = process_pred_mask(pred_mask)

        image_out_folder = os.path.join(path, data_type, f"{idx}")
        create_dir_if_not_exists(image_out_folder)

        image_mask_unet = apply_mask_in_image(test_img, pred_mask)
        image_mask_original = apply_mask_in_image(test_img, numpy.uint8(test_mask[0, :, :, 0] > 0.5))
        change_color_background(image_mask_unet)
        change_color_background(image_mask_original)

        save_image_original(idx, image_out_folder, test_img)
        save_mask_original(idx, image_out_folder, test_mask)
        save_mask_unet(idx, image_out_folder, image_pred_mask)
        save_image_with_mask_original(idx, image_mask_original, image_out_folder)
        save_image_with_mask_unet(idx, image_mask_unet, image_out_folder)
        save_image_post_pred_mask(idx, image_out_folder, post_pred_mask)

        save_image_mask_predmask(os.path.join(path, f"{idx}.png"), test_img[0, :, :, 0], test_mask[0, :, :, 0],
                                 pred_mask, post_pred_mask, image_mask_original, image_mask_unet)


def save_image_post_pred_mask(idx, image_out_folder, post_pred_mask):
    skimage.io.imsave(os.path.join(image_out_folder, f"{idx}-post-pred-mask.png"), post_pred_mask * 255)


def get_image_by_idx(idx, x, y):
    return x[idx, :, :, :].reshape((1, image_size, image_size, 1)), y[idx, :, :, :].reshape(
        (1, image_size, image_size, 1))


def save_image_with_mask_original(idx, image_mask_original, image_out_folder):
    skimage.io.imsave(os.path.join(image_out_folder, f"{idx}-image-mask-original.png"),
                      skimage.img_as_ubyte(image_mask_original))


def save_image_with_mask_unet(idx, image_mask_unet, image_out_folder):
    skimage.io.imsave(os.path.join(image_out_folder, f"{idx}-image-mask-unet.png"),
                      skimage.img_as_ubyte(image_mask_unet))


def save_mask_unet(idx, image_out_folder, image_pred_mask):
    skimage.io.imsave(os.path.join(image_out_folder, f"{idx}-mask-unet.png"), skimage.img_as_ubyte(image_pred_mask))


def save_mask_original(idx, image_out_folder, test_mask):
    skimage.io.imsave(os.path.join(image_out_folder, f"{idx}-mask-original.png"),
                      skimage.img_as_ubyte(test_mask[0, :, :, 0]))


def save_image_original(idx, image_out_folder, test_img):
    skimage.io.imsave(os.path.join(image_out_folder, f"{idx}-original.png"),
                      skimage.img_as_ubyte(test_img[0, :, :, 0]))


def save_all_images(x_test, x_train, x_val, y_test, y_train, y_val, model, path):
    thread_train = threading.Thread(target=save_images, args=(x_train, y_train, "train", model, path,))
    thread_val = threading.Thread(target=save_images, args=(x_val, y_val, "val", model, path,))
    thread_test = threading.Thread(target=save_images, args=(x_test, y_test, "test", model, path,))
    thread_train.start()
    thread_val.start()
    thread_test.start()


def load_data(x, y, path, type_data):
    create_images_used(x, path, type_data)
    x = numpy.array(list([load_image(x["path_image"]) for x in x]))
    y = numpy.array(list([load_mask(y["path_mask"]) for y in y]))
    return format_data(x, y)


def load_all_data(x_train, x_val, x_test, path):
    x_train, y_train = load_data(x_train, numpy.copy(x_train), path, "train")
    x_val, y_val = load_data(x_val, numpy.copy(x_val), path, "val")
    x_test, y_test = load_data(x_test, numpy.copy(x_test), path, "test")
    return x_train, y_train, x_val, y_val, x_test, y_test


def get_all_images_and_masks():
    list_images_mask = get_path_all_images()
    x = numpy.array([{"path_image": file["path_image"], "path_mask": file["path_mask"]} for file in list_images_mask])
    y = numpy.array([file["label"] for file in list_images_mask])
    x, y = sklearn.utils.shuffle(x, y, random_state=1234)
    return x, y


def get_xy_val(x_train, y_train, cfg):
    return sklearn.model_selection.train_test_split(x_train, y_train, test_size=cfg["val_size"], random_state=1234)
