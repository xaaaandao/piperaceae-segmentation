import tensorflow


def calculate_iou_dice(x_test, x_train, x_val, y_test, y_train, y_val, model):
    iou_train, dice_train = model.evaluate(x_train, y_train, verbose=False)
    iou_val, dice_val = model.evaluate(x_val, y_val, verbose=False)
    iou_test, dice_test = model.evaluate(x_test, y_test, verbose=False)
    return {"dice_test": dice_test, "dice_train": dice_train, "dice_val": dice_val, "iou_test": iou_test,
            "iou_train": iou_train, "iou_val": iou_val}


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


def sum_all_metrics(sum_metrics, metrics):
    for key, value in sum_metrics.items():
         sum_metrics.update({key: metrics[key] + value})


def mean_all_metrics(metrics, fold):
    for key, value in metrics.items():
        metrics.update({key: value / fold})
