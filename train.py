from time import strftime, gmtime, localtime
from typing import Any, Callable
import tensorflow as tf

from model.cnn_lstm import SimpleCNNLSTM, MobileNetV3LSTM
from tf_records import parse_tfrecord
import cv2
import numpy as np
import os
import glob
import random
import argparse
from utils import *


class Augment(tf.keras.layers.Layer):
    class SquashSequence(tf.keras.layers.Layer):
        def __init__(self, shape: tuple[int, int, int, int]):
            super().__init__()
            self._shape = shape

        def build(self, input_shape):
            pass

        def call(self, s, *args, **kwargs):
            s = tf.transpose(s, (1, 2, 0, 3))
            return tf.reshape(s, (self._shape[1], self._shape[2], self._shape[0] * self._shape[3]))

    class UnSquashSequence(tf.keras.layers.Layer):
        def __init__(self, shape: tuple[int, int, int, int]):
            super().__init__()
            self._shape = shape

        def build(self, input_shape):
            pass

        def call(self, s, *args, **kwargs):
            s = tf.reshape(s, (self._shape[1], self._shape[2], self._shape[0], self._shape[3]))
            return tf.transpose(s, (2, 0, 1, 3))

    def __init__(self, shape: tuple[int, int, int, int], rescale: Callable):
        super().__init__()
        self._shape = shape
        self._resize_and_rescale = tf.keras.Sequential([
            tf.keras.layers.Resizing(*self._shape[1:3]),
            rescale
        ])

    def call(self, inputs, labels):
        inputs = self.augment_inputs(inputs)
        # labels = self.augment_labels(labels)
        return inputs, labels


class AugmentTrain(Augment):
    def __init__(self,
                 shape: tuple[int, int, int, int],
                 rescale: Callable,
                 output_range: tuple[float, float]):
        super().__init__(shape, rescale)
        self.augment_inputs = tf.keras.Sequential([
            Augment.SquashSequence(shape),
            tf.keras.layers.RandomContrast(factor=0.25),
            self._resize_and_rescale,
            tf.keras.layers.RandomTranslation(height_factor=0.1, width_factor=.01,
                                              fill_mode="constant"),
            tf.keras.layers.RandomZoom(height_factor=0.1, width_factor=.01,
                                       fill_mode="constant"),
            tf.keras.layers.RandomRotation(.05, fill_mode="constant"),
            tf.keras.layers.RandomBrightness(factor=[-0.25, +0.25], value_range=output_range),
            Augment.UnSquashSequence(shape)
        ])


class AugmentValidate(Augment):
    def __init__(self, shape: tuple[int, int, int, int], rescale: Callable):
        super().__init__(shape, rescale)
        self.augment_inputs = tf.keras.Sequential([
            Augment.SquashSequence(shape),
            self._resize_and_rescale,
            Augment.UnSquashSequence(shape)
        ])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=32)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--buffer-size', type=int, default=64)
    parser.add_argument('--show-train-dataset', type=bool, default=False)
    parser.add_argument('--weights-file-name', type=str, default=None)
    args = parser.parse_args()

    root_dir = None
    if os.path.isdir("c:"):
        root_dir = "C:\\Users\\gunte\\OneDrive\\Desktop\\Projects\\BrakeLightDetection"
    elif os.path.isdir("/mnt/c"):
        root_dir = "/mnt/c/Users/gunte/OneDrive/Desktop/Projects/BrakeLightDetection"

    if root_dir is None or not os.path.isdir(root_dir):
        print(f"Unable to find {root_dir}.")
        exit(-1)

    model_dir = os.path.join(root_dir, "data", "trained_models")
    os.makedirs(model_dir, exist_ok=True)

    tf_records_file_path_train = os.path.join(root_dir, "data", "tf_records", "train.tf_record")
    tf_records_file_path_valid = os.path.join(root_dir, "data", "tf_records", "valid.tf_record")

    shape = (32, 128, 128, 3)
    num_units = 1024
    num_classes = max(SequenceTypes.types, key=lambda seq: seq.classification).classification + 1
    nn = SimpleCNNLSTM(shape, num_units, num_classes)
    # nn = MobileNetV3LSTM(shape, num_units, num_classes)
    nn.model().summary()

    checkpoints_dir = os.path.join(
        model_dir,
        "checkpoints",
        f"{strftime('%Y-%m-%d-%H-%M', localtime())}" + "-{val_loss:.2f}.tf"
    )

    if args.weights_file_name is not None:
        weights_file_name = os.path.join(model_dir, "checkpoints", args.weights_file_name)
        if os.path.isfile(weights_file_name + ".index"):
            try:
                nn.model().load_weights(weights_file_name)
                print(f"Loaded weights file: {weights_file_name}")
            except (Exception, ):
                pass

    if args.epochs or args.show_train_dataset:
        train_dataset = load_tfrecord_dataset(tf_records_file_path_train, shape, num_classes)
        train_dataset = train_dataset.shuffle(buffer_size=args.buffer_size)
        train_dataset = train_dataset.map(AugmentTrain(shape, nn.preprocess_input(), nn.preprocessed_range()))
        train_dataset = train_dataset.batch(batch_size=args.batch_size)

        validate_dataset = load_tfrecord_dataset(tf_records_file_path_valid, shape, num_classes)
        validate_dataset = validate_dataset.shuffle(buffer_size=args.buffer_size)
        validate_dataset = validate_dataset.map(AugmentValidate(shape, nn.preprocess_input()))
        validate_dataset = validate_dataset.batch(batch_size=args.batch_size)

        if args.show_train_dataset:
            for Xb, yb in train_dataset:
                for b in range(Xb.shape[0]):
                    X = Xb[b].numpy()
                    for i, image in enumerate(X):
                        image = nn.restore_output_to_cv(image)
                        y = yb[b].numpy().astype(np.int32)
                        # y = np.argmax(y)
                        y = ClassificationNames[y]
                        image_draw = image.copy()
                        image_draw = cv.putText(image_draw, y, (20, 20), cv.FONT_HERSHEY_PLAIN, 1.0,
                                                (255, 255, 255), 1)
                        cv.namedWindow(f"tfr", cv.WINDOW_NORMAL)
                        cv.resizeWindow(f"tfr", 64, 64)
                        cv.imshow(f"tfr", image_draw)

                        cv.waitKey(33)
                        pass

        if args.epochs:
            callbacks = [
                tf.keras.callbacks.ModelCheckpoint(checkpoints_dir,
                                                   monitor='val_loss',
                                                   save_best_only=True,
                                                   save_weights_only=True),
            ]

            optimizer = tf.keras.optimizers.Adam(learning_rate=.00001)
            nn.model().compile(optimizer=optimizer,
                          loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                          metrics=['accuracy'])
            nn.model().fit(train_dataset,
                      epochs=args.epochs,
                      validation_data=validate_dataset,
                      callbacks=callbacks)


if __name__ == '__main__':
    main()
