import numpy
import pathlib
import skimage
import skimage.io

import cv2


def return_genus(nome_arquivo):
    lista_generos = list(["manekia", "ottonia", "piper", "pothomorphe", "peperomia"])
    return next((genero for genero in lista_generos if genero in nome_arquivo), None)


def carrega_mascara(nome_arquivo):
    return numpy.float32(skimage.io.imread(nome_arquivo) / 255)


def retorna_todas_mascaras():
    return zip(*[(carrega_mascara(nome_arquivo), return_genus(nome_arquivo.name))
                 for nome_arquivo in sorted(pathlib.Path("images/mascara").rglob("*"))])


def carrega_imagem_original(nome_arquivo):
    return skimage.img_as_float32(skimage.io.imread(nome_arquivo))


def retorna_todas_imgs_originais():
    return [carrega_imagem_original(nome_arquivo) for nome_arquivo in sorted(pathlib.Path("images/originais").rglob("*"))]


def main():
    lista_todas_mascaras, lista_generos = retorna_todas_mascaras()
    lista_todas_imgs_originais = retorna_todas_imgs_originais()
    print(lista_todas_mascaras, type(lista_todas_mascaras), len(lista_todas_mascaras))
    print(lista_generos, type(lista_generos), len(lista_generos))
    print(lista_todas_imgs_originais, type(lista_todas_imgs_originais), len(lista_todas_imgs_originais))


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
