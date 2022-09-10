import numpy
import tensorflow


class AugmentationSequence(tensorflow.keras.utils.Sequence):
    def __init__(self, x_set, y_set, batch_size, augmentations):
        self.x, self.y = x_set, y_set
        self.batch_size = batch_size
        self.augment = augmentations
        print(f"b{self.x}")

    def __len__(self):
        print(f"b{int(numpy.ceil(len(self.x) / float(self.batch_size)))}")
        return int(numpy.ceil(len(self.x) / float(self.batch_size)))

    def __getitem__(self, idx):
        print(f"c{idx}")

        batch_x = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]

        aug_x = numpy.zeros(batch_x.shape)
        aug_y = numpy.zeros(batch_y.shape)

        for idx in range(batch_x.shape[0]):
            aug = self.augment(image=batch_x[idx, :, :, :], mask=batch_y[idx, :, :, :])
            aug_x[idx, :, :, :] = aug["image"]
            aug_y[idx, :, :, :] = aug["mask"]

        return aug_x, aug_y