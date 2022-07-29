import numpy


def pred_mask(image, model):
    # mask = model.predict(image)[0,:,:,0]
    # return numpy.uint8(mask > 0.5)
    return model.predict(image)[0,:,:,0]


def apply_mask(image, mask):
    return image[0,:,:,0] * mask


def convert_background_color(image):
    image[image == 0] = 1
