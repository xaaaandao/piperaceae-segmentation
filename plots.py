import matplotlib.pyplot


def plot_lossgraph(filename, model):
    figure, axis = matplotlib.pyplot.subplots(1, figsize=(10, 10))
    matplotlib.pyplot.ioff()
    axis.plot(model.history["loss"], label="Train")
    axis.plot(model.history["val_loss"], label="Validation")
    axis.plot(model.history["lr"], label="Learning rate")
    figure.suptitle("Train, Validation and Learning Rate", fontsize=20, verticalalignment="center")
    axis.set_ylabel("Loss", fontsize=16)
    axis.set_xlabel("Epoch", fontsize=16)
    axis.legend()
    figure.savefig(filename)
    matplotlib.pyplot.cla()
    matplotlib.pyplot.clf()
    matplotlib.pyplot.close()
