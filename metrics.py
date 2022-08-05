import dataclasses
import tensorflow


def calculate_iou_dice(model, x_test, x_train, x_val, y_test, y_train, y_val):
    print(model.metrics_names)
    loss_train, dice_train, jaccard_train, precision_train, recall_train =\
        model.evaluate(x_train, y_train, verbose=False)
    loss_val, dice_val, jaccard_val, precision_val, recall_val =\
        model.evaluate(x_val, y_val, verbose=False)
    loss_test, dice_test, jaccard_test, precision_test, recall_test =\
        model.evaluate(x_test, y_test, verbose=False)
    return Metrics(dice_test, dice_train, dice_val, jaccard_test, jaccard_train, jaccard_val, loss_test, loss_train, loss_val, precision_test, precision_train, precision_val, recall_test, recall_train, recall_val)


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


@dataclasses.dataclass
class Metrics:
    dice_test: float
    dice_train: float
    dice_val: float
    jaccard_test: float
    jaccard_train: float
    jaccard_val: float
    loss_test: float
    loss_train: float
    loss_val: float
    precision_test: float
    precision_train: float
    precision_val: float
    recall_test: float
    recall_train: float
    recall_val: float