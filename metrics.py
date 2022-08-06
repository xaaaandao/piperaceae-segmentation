import tensorflow


# Loss functions
def jaccard_distance_loss(y_true, y_pred, smooth=1):
    intersection = tensorflow.math.reduce_sum(tensorflow.math.abs(y_true * y_pred))
    union = tensorflow.math.reduce_sum(tensorflow.math.abs(y_true) + tensorflow.math.abs(y_pred))
    jac = (intersection + smooth) / (union - intersection + smooth)
    loss = (1 - jac)
    return loss


def dice_loss(y_true, y_pred, smooth=1):
    intersection = tensorflow.math.reduce_sum(tensorflow.math.abs(y_true * y_pred))
    union = tensorflow.math.reduce_sum(tensorflow.math.abs(y_true) + tensorflow.math.abs(y_pred))
    dice = (2. * intersection + smooth) / (union + smooth)
    loss = 1 - dice
    return loss


def jaccard_distance(y_true, y_pred, smooth=1):
    intersection = tensorflow.math.reduce_sum(tensorflow.math.abs(y_true * y_pred))
    union = tensorflow.math.reduce_sum(tensorflow.math.abs(y_true) + tensorflow.math.abs(y_pred))
    return (intersection + smooth) / (union - intersection + smooth)


def dice_coef(y_true, y_pred, smooth=1):
    intersection = tensorflow.math.reduce_sum(tensorflow.math.abs(y_true * y_pred))
    union = tensorflow.math.reduce_sum(tensorflow.math.abs(y_true) + tensorflow.math.abs(y_pred))
    return (2. * intersection + smooth) / (union + smooth)
