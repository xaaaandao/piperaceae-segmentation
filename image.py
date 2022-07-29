def pred_mask(image, model):
    return model.predict(image)[0,:,:,0]


def apply_mask(image, mask):
    return image[0,:,:,0] * mask


def convert_background_color(image):
    image[image == 0] = 1
