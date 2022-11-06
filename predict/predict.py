import click
import os
import pathlib
import sys
import tensorflow as tf

from PIL import Image

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


from metrics import dice_coef, dice_loss, jaccard_distance


def image_is_rgb(color_mode):
    return color_mode == 3


def image_is_grayscale(color_mode):
    return color_mode == 1


def get_path_best_model_rgb_and_size_is_256():
    return '/home/xandao/Documentos/resultados_gimp/u-net/RGB/23-10-2022-22-39-56/0/unet.h5'


def get_path_best_model_rgb_and_size_is_400():
    return '/home/xandao/Documentos/resultados_gimp/u-net/RGB/23-10-2022-22-14-17/3/unet.h5'


def get_path_best_model_rgb_and_size_is_512():
    return '/home/xandao/Documentos/resultados_gimp/u-net/RGB/23-10-2022-22-52-42/4/unet.h5'


def get_path_best_model_grayscale_and_size_is_256():
    return '/home/xandao/Documentos/resultados_gimp/u-net/grayscale/23-10-2022-20-19-47/3/unet.h5'


def get_path_best_model_grayscale_and_size_is_400():
    return '/home/xandao/Documentos/resultados_gimp/u-net/grayscale/23-10-2022-21-31-58/3/unet.h5'


def get_path_best_model_grayscale_and_size_is_512():
    return '/home/xandao/Documentos/resultados_gimp/u-net/grayscale/23-10-2022-20-55-01/4/unet.h5'


def get_dir_out_grayscale(image_size, taxon, threshold):
    return f'/home/xandao/Documentos/GitHub/dataset_gimp/imagens_george/imagens/grayscale/{taxon}/{image_size}/{threshold}/'


def get_dir_out_rgb(image_size, taxon, threshold):
    return f'/home/xandao/Documentos/GitHub/dataset_gimp/imagens_george/imagens/RGB/{taxon}/{image_size}/{threshold}/'


def image_size_is_256(image_size):
    return image_size == 256


def image_size_is_400(image_size):
    return image_size == 400


def image_rgb_size_is_not_256(image_size):
    return get_path_best_model_rgb_and_size_is_400() if image_size_is_400(image_size) else get_path_best_model_rgb_and_size_is_512()


def image_grayscale_size_is_not_256(image_size):
    return get_path_best_model_rgb_and_size_is_400() if image_size_is_400(image_size) else get_path_best_model_rgb_and_size_is_512()


def get_model_rgb(image_size):
    return get_path_best_model_rgb_and_size_is_256() if image_size_is_256(image_size) else image_rgb_size_is_not_256(image_size)


def get_model_grayscale(image_size):
    return get_path_best_model_grayscale_and_size_is_256() if image_size_is_256(image_size) else image_grayscale_size_is_not_256(image_size)


def get_unet_model(color_mode, image_size):
    return get_model_rgb(image_size) if image_is_rgb(color_mode) else get_model_grayscale(image_size)


def get_dir_out(color_mode, image_size, taxon, threshold):
    return get_dir_out_rgb(image_size, taxon, threshold) if image_is_rgb(color_mode) else get_dir_out_grayscale(image_size, taxon, threshold)


def create_dir_out(list_f):
    for f in list_f:
        for d in ['mask_unet', 'w_pred_mask', 'transparency']:
            p = os.path.join(f, d)
            print(p)
            pathlib.Path(p).mkdir(parents=True, exist_ok=True)


def save_image_segmented_background_transparency(file, image_original, mask, path):
    mask = tf.keras.preprocessing.image.array_to_img(mask).convert('L')
    image_original = image_original.convert('RGBA')
    filename = os.path.join(path, 'transparency', str(file.stem) + '_transparente.png')
    print(filename)
    image_original.putalpha(mask)
    image_original.save(filename)
    return image_original


def get_image(color_mode, file):
    return tf.keras.preprocessing.image.load_img(file.resolve(), color_mode='grayscale') if image_is_grayscale(color_mode) else tf.keras.preprocessing.image.load_img(file.resolve())


def set_color_mode_background(background, color_mode):
    return background.convert('L') if image_is_grayscale(color_mode) else background.convert('RGB')


def save_image_segmented_background_white(color_mode, file, image_original, image_size, path):
    background = Image.new('RGBA', (image_size, image_size), 'WHITE')
    image_original_width, image_original_height = image_original.size
    background_width, background_height = background.size
    offset = ((background_width - image_original_width) // 2, (background_height - image_original_height) // 2)
    background.paste(image_original, offset, image_original)
    filename = os.path.join(path, 'w_pred_mask', str(file.stem) + '.jpeg')
    background = set_color_mode_background(background, color_mode)
    background.save(filename)


def save_mask(color_mode, file, image_original, image_size, model, path):
    image = tf.keras.preprocessing.image.img_to_array(image_original)
    image = image / 255
    image = image.reshape((1, image_size, image_size, color_mode))
    mask = model.predict(image)
    mask = mask[0, :, :, :]
    filename = os.path.join(path, 'mask_unet', str(file.stem) + '.bmp')
    print(filename)
    tf.keras.preprocessing.image.save_img(filename, mask)
    return mask


@click.command()
@click.option(
    '--color',
    '-c',
    type=click.Choice(['RGB', 'grayscale']),
    required=True
)
@click.option(
    '--size',
    '-s',
    multiple=True,
    default=[1],
    help='Image size',
    type=int,
    required=True
)
@click.option(
    '--taxon',
    '-t',
    type=click.Choice(['genus', 'specific_epithet']),
    required=True
)
@click.option(
    '--threshold',
    multiple=True,
    default=[1],
    type=int,
    required=True
)
def main(color, size, taxon, threshold):
    for image_size in size:
        for threshold in threshold:
            unet_model = get_unet_model(color, image_size)
            dir_out = get_dir_out(color, image_size, taxon, threshold)
            list_images = sorted([file for file in pathlib.Path(dir_out).rglob('*.jpeg')])
            list_dir_out = sorted([d for d in pathlib.Path(dir_out).glob('*') if d.is_dir()])
            create_dir_out(list_dir_out)
            model = tf.keras.models.load_model(unet_model, custom_objects = {'dice_loss': dice_loss, 'dice_coef': dice_coef, 'jaccard_distance': jaccard_distance })

            for i, file in enumerate(list_images):
                print(i, file.resolve())
                image_original = get_image(color, file)
                index_path = str(file.resolve()).index('jpeg')
                path = str(file.resolve())[0:index_path]
                mask = save_mask(color, file, image_original, image_size, model, path)
                image_original = save_image_segmented_background_transparency(file, image_original, mask, path)
                save_image_segmented_background_white(color, file, image_original, image_size, path)


if __name__ == '__main__':
    main()
