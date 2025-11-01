import click
import os
import pathlib


from PIL import Image

# input_path = "/mnt/eec07521-c36a-4d2b-9047-0110e7749eae/Nextcloud/Dropbox import/datasets/2025/images/iwssip/images/sem_resize"
@click.command()
@click.option("--img_size", type=int)
@click.option("--input_dir", type=str)
@click.option("--output_dir", type=str)
def main(img_size, input_dir, output_dir):
    for i, p in enumerate(pathlib.Path(input_dir).rglob("*")):
        print(i, p)
        # for img_shape in [(32, 32), (64, 64), (128, 128), (256, 256), (400, 400), (512, 512)]:
        # output_path = "/mnt/eec07521-c36a-4d2b-9047-0110e7749eae/Nextcloud/Dropbox import/datasets/2025/images/iwssip/images/%d" % img_shape[0]
        img = Image.open(p)
        os.makedirs(output_dir, exist_ok=True)
        filename = os.path.join(output_dir, "%s.bmp" % p.stem)
        if not os.path.exists(filename):
            img = img.resize((img_size, img_size))
            img.save(filename)
            print("saving %s" % filename)

if __name__ == "__main__":
    main()