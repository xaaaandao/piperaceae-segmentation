import os
import tensorflow


def set_avx_avx2():
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'  # INFO messages are not printed


def set_gpu():
    gpus = tensorflow.config.experimental.list_physical_devices("GPU")
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                print(f"GPU: {gpu.name}")
                tensorflow.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)
