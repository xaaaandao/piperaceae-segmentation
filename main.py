# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import pathlib
import cv2
import os
import tensorflow
import unet


def main():
    # Use a breakpoint in the code line below to debug your script.
    # print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.
    # gpus = tensorflow.config.experimental.list_physical_devices("GPU")
    # print(gpus)

    todas_imagens = []
    rotulos = []
    pasta_imgs = 'images/originais'
    pasta_mascara = 'images/mascara'
    for arquivo in sorted(os.listdir(pasta_imgs)):
        print(arquivo)
        imagem = cv2.imread(os.pathjoin(pasta_imgs, arquivo))
        imagem = cv2.img_as_float32(imagem)
        todas_imagens.append(imagem)
        #
        #
        imagem_mascara = cv2.imread(os.pathjoin(pasta_mascara, arquivo))
        imagem_mascara = cv2.img_as_float32(imagem_mascara/255)

        # rotulos.append(imagem_mascara)
        # classes.append(1)

    model = unet.unet_model()
    # adam_opt = tensorflow.keras.optimizers.Adam(learning_rate = 0.001)
    fit = model.fit(X_treinamento, epochs=100, validation_data=(X_teste, y_teste))
    # model.compile(optimizer = adam_opt, loss = jaccard_distance_loss, metrics = [dice_coef])

    # for img_path in os.listdir(os.path.join(montgomery_folder, "CXR")):
    #     img = imread(os.path.join(montgomery_folder, "CXR", img_path))
    #     img = img_as_float32(img)
    #     montgomery_images.append(img)
    #     images.append(img)
    #
    #     mask = imread(os.path.join(montgomery_folder, "Masks", img_path))
    #     mask = np.float32(mask / 255)
    #     montgomery_labels.append(mask)
    #     labels.append(mask)
    #
    # len(montgomery_labels)

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
