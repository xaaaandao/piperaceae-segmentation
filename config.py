import os

import tensorflow


def define_avx_avx2():
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def define_gpu():
    gpus = tensorflow.config.experimental.list_physical_devices("GPU")
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                print(f"A gpu é: {gpu.name}")
                tensorflow.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)
