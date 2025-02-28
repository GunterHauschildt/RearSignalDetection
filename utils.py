import numpy as np
from numpy.typing import NDArray
import cv2 as cv
import glob
from copy import copy
from dataclasses import dataclass
import tensorflow as tf


def parse_tfrecord(tfrecord, output_shape: tuple[int, int, int, int], num_classes):
    features = tf.io.parse_single_example(tfrecord, features={
        'sequence_length': tf.io.FixedLenFeature([], tf.int64),
        'height'         : tf.io.FixedLenFeature([], tf.int64),
        'width'          : tf.io.FixedLenFeature([], tf.int64),
        'channels'       : tf.io.FixedLenFeature([], tf.int64),
        'classification' : tf.io.FixedLenFeature([], tf.int64),
        'sequence'       : tf.io.FixedLenFeature([], tf.string)
    })

    sequence_length = tf.cast(features['sequence_length'], tf.int64)
    height = tf.cast(features['height'], tf.int64)
    width = tf.cast(features['width'], tf.int64)
    channels = tf.cast(features['channels'], tf.int64)
    classification = tf.cast(features['classification'], tf.int64)
    sequence = tf.io.decode_raw(features['sequence'], tf.uint8)
    sequence_shape = [sequence_length, height, width, channels]
    sequence = tf.reshape(sequence, sequence_shape)

    # we do the resizing here. but there's still alot of squashing. unsquashing to come!
    # ultimately it seems the only 'non-sized' tf function is resizing, which we can't call
    # first. For any other augmentation while training, we need to squash and unsquash again

    sequence = tf.transpose(sequence, (1, 2, 0, 3))
    sequence = tf.reshape(sequence, (
        height, width, sequence_length * channels
    ))
    sequence = tf.image.resize(sequence, (output_shape[1:3]))
    sequence = tf.reshape(sequence, (
        output_shape[1], output_shape[2], sequence_length, channels
    ))
    
    # we also do the BGR to RGB here
    sequence = sequence[:, :, :, ::-1]
    sequence = tf.transpose(sequence, (2, 0, 1, 3))

    return sequence, classification


def load_tfrecord_dataset(tf_records_file_path, shape: tuple[int, int, int, int], num_classes: int):
    tf_records = tf.data.Dataset.list_files(tf_records_file_path)
    dataset = tf_records.flat_map(tf.data.TFRecordDataset)
    return dataset.map(lambda x: parse_tfrecord(x, shape, num_classes))


ClassificationNames = {
    0: "No Signal",
    1: "Left Turn",
    2: "Right Turn",
    3: "Hazard"
}


@dataclass
class SequenceType:
    dataset_name: str
    classification: int
    name: str


@dataclass
class SequenceTypes:
    types = [SequenceType("OOO", 0, ClassificationNames[0]),
             SequenceType("BOO", 0, ClassificationNames[0]),
             SequenceType("OLO", 1, ClassificationNames[1]),
             SequenceType("BLO", 1, ClassificationNames[1]),
             SequenceType("OOR", 2, ClassificationNames[2]),
             SequenceType("BOR", 2, ClassificationNames[2]),
             SequenceType("OLR", 3, ClassificationNames[3]),
             SequenceType("BLR", 3, ClassificationNames[3])]

@dataclass
class ImageSequence:
    classification: int
    classification_name: str
    sequence_name: str
    folder_name: str
    ext: str


def image_sequence_folder_to_numpy(folder_name: str,
                                   sequence_length: int,
                                   shape: tuple[int, int] or None = None,
                                   classification_name: str or None = None,
                                   draw: bool = False,
                                   wait: int = 1,
                                   ) -> NDArray | None:

    print(f"{folder_name}")

    image_files = glob.glob(folder_name + "\*.png")

    if not len(image_files):
        return None
    recorded_image_files = copy(image_files)

    # grow by adding the reversed sequence until its long enough
    while len(image_files) < sequence_length:
        recorded_image_files.reverse()
        image_files += recorded_image_files[1:]
    image_files = image_files[:sequence_length]

    images = [cv.imread(image_file) for image_file in image_files]
    if shape is None:
        w = max(image.shape[1] for image in images)
        h = max(image.shape[0] for image in images)
    else:
        w = shape[1]
        h = shape[0]

    a = []
    for image in images:
        image = cv.resize(image, (w, h))
        if draw:
            image_draw = image.copy()
            if classification_name is not None:
                image_draw = cv.putText(
                    image_draw,
                    f"{classification_name}",
                    (20, 20), cv.FONT_HERSHEY_PLAIN, 1.0, (255, 255, 255), 2
                )
            win_name = "Seq2Numpy"
            cv.namedWindow(win_name, cv.WINDOW_AUTOSIZE)
            cv.imshow(win_name, image_draw)
            cv.waitKey(wait)

        a.append(image)
    return np.array(a)


def image_sequence_folder_to_list(folder_name: str) -> list | None:

    print(f"{folder_name}")

    image_files = glob.glob(folder_name + "\*.png")

    if not len(image_files):
        return None

    return [cv.imread(image_file) for image_file in image_files]




