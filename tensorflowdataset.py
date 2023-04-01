import os 
import matplotlib.pyplot

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf
from tensorflow import keras
from tensorflow.python.keras.layers import Dense
import tensorflowdataset as tfds

physical_devices = tf.config.list_physical_devices("GPU")
tf.config.experimental.set_memory_growth(physical_devices[0], True)

(ds_train, ds_test), ds_info = tfds.load(
    "mnist",
    split = ["train", "test"], 
    shuffle_files = True,
    as_supervised = True,
    with_info = True,
)

fig = tfds.show_examples(ds_train, ds_info, rows = 4, cols = 4)
print(ds_info)

