import click
import os
import pathlib
import tensorflow as tf

from PIL import Image
from metrics import dice_coef, dice_loss, jaccard_distance


def save_image_segmented_background_transparency(file, image_original, mask, path):
    mask = tf.keras.preprocessing.image.array_to_img(mask).convert("L")
    image_original = image_original.convert("RGBA")
    filename = os.path.join(path, "transparency", "%s_transparente.png" % file.stem)
    print("saving %s" % filename)
    image_original.putalpha(mask)
    image_original.save(filename)
    return image_original


def save_image_segmented_background_white(file, image_original, image_size, path):
    background = Image.new("RGBA", (image_size, image_size), "WHITE")
    image_original_width, image_original_height = image_original.size
    background_width, background_height = background.size
    offset = ((background_width - image_original_width) // 2, (background_height - image_original_height) // 2)
    background.paste(image_original, offset, image_original)
    filename = os.path.join(path, "w_pred_mask", "%s.jpeg" % file.stem)
    print("saving %s" % filename)
    background = background.convert("RGB")
    background.save(filename)


def save_mask(file, image_original, image_size, model, path):
    image = tf.keras.preprocessing.image.img_to_array(image_original)
    image = image / 255
    print(image.shape)
    image = image.reshape((1, image_size, image_size, 3))
    mask = model.predict(image)
    mask = mask[0, :, :, :]
    filename = os.path.join(path, "mask_unet", "%s.bmp" % file.stem)
    tf.keras.preprocessing.image.save_img(filename, mask)
    print("saving %s" % filename)
    return mask


@click.command()
@click.option("--best", type=str)
@click.option("--img_size", type=int)
@click.option("--input_dir", type=str)
@click.option("--output_dir", type=str)
def main(best, img_size, input_dir, output_dir):
    for d in ["mask_unet", "w_pred_mask", "transparency"]:
        output = os.path.join(output_dir, d)
        os.makedirs(output, exist_ok=True)
        

    images = sorted([file for file in pathlib.Path(input_dir).rglob("*.jpeg")])
    model = tf.keras.models.load_model(best, custom_objects = {"dice_loss": dice_loss, "dice_coef": dice_coef, "jaccard_distance": jaccard_distance })

    for i, image in enumerate(images):
        print(i, image.resolve())
        # se for escala de cinza colocar color_mode="grayscale"
        image_original = tf.keras.preprocessing.image.load_img(image.resolve())
        index_path = str(image.resolve()).index("jpeg")
        path = str(image.resolve())[0:index_path]
        # mask = save_mask(image, image_original, img_size, model, path)
        # image_original = save_image_segmented_background_transparency(image, image_original, mask, path)
        # save_image_segmented_background_white(image, image_original, img_size, path)


if __name__ == "__main__":
    main()
