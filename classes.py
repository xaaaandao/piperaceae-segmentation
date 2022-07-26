import dataclasses
import enum
import numpy
import os
import pathlib
import tensorflow


from files import load_mask, load_image

class Genus(enum.Enum):
    MANEKIA = "manekia"
    OTTONIA = "ottonia"
    PEPEROMIA = "peperomia"
    PIPER = "piper"
    POTHOMORPHE = "pothomorphe"


class TypeImage(enum.Enum):
    ORIGINAL = "original"
    MASK = "mask"


@dataclasses.dataclass(init=True)
class Image:
    cfg: dict
    file: pathlib
    image: numpy.ndarray = dataclasses.field(init=False)
    label: Genus = dataclasses.field(init=False)
    height: int = dataclasses.field(init=False, default=0)
    width: int = dataclasses.field(init=False, default=0)
    type: TypeImage

    def __post_init__(self):
        self.set_size()
        setattr(self, "label", self.get_label())
        self.image = self.load_image()

    def set_size(self):
        height, width = self.get_height_and_width_of_image()

        if self.cfg["image_size"] != height or self.cfg["image_size"] != width:
            raise ValueError("value of image_size different")

        setattr(self, "height", height)
        setattr(self, "width", width)

    def get_label(self):
        list_genus = (Genus.MANEKIA, Genus.OTTONIA, Genus.PEPEROMIA, Genus.PIPER, Genus.POTHOMORPHE)
        return next((g for g in list_genus if g.value in str(self.file.stem)))

    def get_height_and_width_of_image(self):
        image = tensorflow.keras.preprocessing.image.load_img(str(self.file.resolve()))
        return image.height, image.width

    def load_image(self):
        return load_mask(str(self.file.resolve())) if getattr(self, "type") else load_image(str(self.file.resolve()))


@dataclasses.dataclass
class Dataset:
    cfg: dict
    list_images: numpy.ndarray = dataclasses.field(init=False)
    path_images: str = dataclasses.field(default="")
    path_mask: str = dataclasses.field(default="")

    def __post_init__(self):
        if not os.path.exists(self.path_images) or not os.path.exists(self.path_mask):
            raise ValueError("path is wrong")
        self.list_images = self.load_images()

    def load_images(self):
        return list([{"image": Image(self.cfg, file, TypeImage.ORIGINAL),
                      "mask": Image(self.cfg, self.get_equivalent_image(str(file.stem)), TypeImage.MASK)} for file in
                     pathlib.Path(self.path_images).rglob("*") if file.is_file()])

    def get_equivalent_image(self, filename):
        list_images = list([p for p in pathlib.Path(self.path_mask).rglob("*") if p.is_file()])
        list_image_equivalent = list(filter(lambda image: str(image.stem) == filename, list_images))

        if len(list_image_equivalent) > 1:
            raise FileExistsError("duplicates of filename")

        return list_image_equivalent[0]


@dataclasses.dataclass
class Index:
    fold: int
    index_train: list
    index_test: list
