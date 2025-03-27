# export LD_LIBRARY_PATH=$CONDA_PREFIX/lib

import tensorflow as tf
print("TensorFlow version:", tf.__version__)
print("GPU Available:", tf.config.list_physical_devices('GPU'))