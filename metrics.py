import dataclasses
import tensorflow


def calculate_iou_dice(model, x_test, x_train, x_val, y_test, y_train, y_val):
    iou_train, dice_train = model.evaluate(x_train, y_train, verbose=False)
    iou_val, dice_val = model.evaluate(x_val, y_val, verbose=False)
    iou_test, dice_test = model.evaluate(x_test, y_test, verbose=False)
    return Metrics(dice_test, dice_train, dice_val, iou_test, iou_train, iou_val)


def jaccard_distance_loss(y_true, y_pred, smooth=100):
    intersection = tensorflow.keras.backend.sum(tensorflow.keras.backend.abs(y_true * y_pred), axis=-1)
    union = tensorflow.keras.backend.sum(tensorflow.keras.backend.abs(y_true) + tensorflow.keras.backend.abs(y_pred),
                                         axis=-1)
    jac = (intersection + smooth) / (union - intersection + smooth)
    loss = (1 - jac) * smooth
    return loss


def dice_coef(y_true, y_pred, smooth=1):
    intersection = tensorflow.keras.backend.sum(tensorflow.keras.backend.abs(y_true * y_pred), axis=-1)
    union = tensorflow.keras.backend.sum(tensorflow.keras.backend.abs(y_true), -1) + tensorflow.keras.backend.sum(
        tensorflow.keras.backend.abs(y_pred), -1)
    return (2. * intersection + smooth) / (union + smooth)


@dataclasses.dataclass
class Metrics:
    dice_test: float
    dice_train: float
    dice_val: float
    iou_test: float
    iou_train: float
    iou_val: float