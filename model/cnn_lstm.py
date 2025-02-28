import tensorflow as tf
import numpy as np
import cv2 as cv
from numpy.typing import NDArray
from typing import Any, Callable


class SimpleCNNLSTM:
    def __init__(self, shape: tuple[int, int, int, int], num_units: int, num_classes: int):

        self._shape = shape
        self._num_classes = num_classes

        self._cnn = tf.keras.Sequential([

            tf.keras.layers.Conv2D(32, 3, padding='same', activation='relu'),
            tf.keras.layers.Conv2D(32, 3, padding='same', activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPooling2D(),

            tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu'),
            tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPooling2D(),

            tf.keras.layers.Conv2D(128, 3, padding='same', activation='relu'),
            tf.keras.layers.Conv2D(128, 3, padding='same', activation='relu'),
            tf.keras.layers.Conv2D(128, 3, padding='same', activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPooling2D(),

            tf.keras.layers.Conv2D(256, 3, padding='same', activation='relu'),
            tf.keras.layers.Conv2D(256, 3, padding='same', activation='relu'),
            tf.keras.layers.Conv2D(256, 3, padding='same', activation='relu'),
            tf.keras.layers.Conv2D(256, 3, padding='same', activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPooling2D(),

            tf.keras.layers.Flatten()  # GlobalMaxPooling2D()
        ])

        inputs = tf.keras.Input(shape=self._shape)
        outputs = tf.keras.layers.TimeDistributed(self._cnn)(inputs)
        outputs = tf.keras.layers.LSTM(num_units, dropout=.50)(outputs)
        outputs = tf.keras.layers.Dense(num_units // 2, activation="relu", name='lstm_dense_1')(outputs)
        outputs = tf.keras.layers.Dropout(.50)(outputs)
        # outputs = tf.keras.layers.Dense(num_units // 4, activation="relu", name='lstm_dense_2')(outputs)
        # outputs = tf.keras.layers.Dropout(.33)(outputs)
        outputs = tf.keras.layers.Dense(self._num_classes, activation="softmax", name=f'output_' + str(self._num_classes))(outputs)
        print(f"{outputs.shape}")
        self._model = tf.keras.Model(inputs, outputs)

    def model(self) -> tf.keras.Model:
        return self._model

    @staticmethod
    def preprocess_input() -> Callable:
        return tf.keras.layers.Rescaling(1. / 255.)

    @staticmethod
    def preprocessed_range() -> tuple[float, float]:
        return 0., 1.

    @staticmethod
    def restore_output_to_cv(x: Any):
        if tf.is_tensor(x):
            x = x.numpy()
        x = cv.cvtColor(x * 255., cv.COLOR_BGR2RGB).astype(np.uint8)
        return x

    @staticmethod
    def input_cv_to_tf(x):
        x = cv.cvtColor(x, cv.COLOR_BGR2RGB)
        x = x.astype(np.float32) / 255.
        return x


class MobileNetV3LSTM:
    def __init__(self, shape: tuple[int, int, int, int], num_units: int, num_classes: int):
        self._shape = shape
        self._num_classes = num_classes

        self._mobile_net = tf.keras.applications.MobileNetV3Small(
            input_shape=self._shape[1:4],
            include_top=False,
            include_preprocessing=False,
            pooling=None
        )
        self._mobile_net.trainable = True

        self._cnn = tf.keras.Sequential([
            self._mobile_net,
            tf.keras.layers.Flatten()
        ])

        inputs = tf.keras.Input(shape=self._shape)
        outputs = tf.keras.layers.TimeDistributed(self._cnn)(inputs)
        outputs = tf.keras.layers.LSTM(num_units, dropout=.50)(outputs)
        outputs = tf.keras.layers.Dense(num_units // 2, activation="relu", name='lstm_dense_1')(outputs)
        outputs = tf.keras.layers.Dropout(.50)(outputs)
        outputs = tf.keras.layers.Dense(num_classes, activation="softmax", name=f'output_' + str(self._num_classes))(outputs)
        print(f"{outputs.shape}")
        self._model = tf.keras.Model(inputs, outputs)

    def model(self):
        return self._model

    @staticmethod
    def preprocess_input() -> Callable:
        return tf.keras.layers.Rescaling(scale=1./127.5, offset=-1.)

    @staticmethod
    def preprocessed_range() -> tuple[float, float]:
        return -1., 1.

    @staticmethod
    def restore_output_to_cv(x):
        if tf.is_tensor(x):
            x = x.numpy()
        x = cv.cvtColor((x + 1.) * 127.5, cv.COLOR_RGB2BGR).astype(np.uint8)
        return x

    @staticmethod
    def input_cv_to_tf(x):
        x = cv.cvtColor(x, cv.COLOR_BGR2RGB)
        x = x.astype(np.float32) / 127.5 - 1.
        return x
