import os.path
import glob
import random
from numpy import typing
from typing import Callable
from sklearn.model_selection import train_test_split
from collections import defaultdict
from collections import namedtuple
from copy import copy
import argparse
from dataclasses import dataclass
import cv2 as cv
import numpy as np
import tensorflow as tf


@dataclass
class ImageSequence:
    classification: int
    classification_name: str
    sequence_name: str
    folder_name: str
    ext: str


def flip_horizontal(image_filename, augmented_filename):
    image = cv.imread(image_filename)
    image_augmented = cv.flip(image, 1)
    cv.imshow("ORIGINAL", image)
    cv.imshow("AUGMENTED", image_augmented)
    cv.imwrite(augmented_filename, image_augmented)
    cv.waitKey(1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=str, default=None)
    parser.add_argument('--ext', type=str, default=".png")
    args = parser.parse_args()

    difficulty_file_prefixes = ['easy', 'moderate', 'hard']
    difficulty_file_names = [(f, os.path.join(args.data_dir, f + ".txt")) for f in difficulty_file_prefixes]
    all_difficulty_folders = defaultdict(lambda: list())
    for difficulty_file_name in difficulty_file_names:
        with open(difficulty_file_name[1]) as difficulty_file:
            while line := difficulty_file.readline():
                line = line.strip()
                if len(line):
                    all_difficulty_folders[difficulty_file_name[0]].append(line)

    @dataclass
    class AugmentTypes:
        original: str | None
        augmented: str | None
        action: callable

    augment_types = [
        ("OOO", None, None), ("BOO", None, None),
        ("OLO", "OOR", flip_horizontal), ("BLO", "BOR", flip_horizontal),
        ("OOR", "OLO", flip_horizontal), ("BOR", "BLO", flip_horizontal),
        ("OLR", None, None), ("BLR", None, None)]

    @dataclass
    class AugmentFolder:
        sequence_name: str
        folder_name: str
        difficulty_name: str
        augment_type: tuple[str, str, callable]

    augment_folders = []
    for folder_name in os.walk(args.data_dir):
        folder_name = folder_name[0]
        if not folder_name.endswith("light_mask"):
            continue
        sequence_name = folder_name.split(os.sep)[-2]

        for augment_type in augment_types:
            if augment_type[0] in folder_name:
                for difficulty_name, difficulty_folders in all_difficulty_folders.items():
                    if sequence_name in difficulty_folders:
                        augment_folders.append(
                            AugmentFolder(sequence_name, folder_name, difficulty_name, augment_type)
                        )

    for augment_folder in augment_folders:
        if augment_folder.augment_type[1] is None:
            continue

        augmented_folder_name = augment_folder.folder_name.replace(
            augment_folder.augment_type[0], augment_folder.augment_type[1]
        )
        os.makedirs(augmented_folder_name, exist_ok=True)
        sequence_name = augment_folder.sequence_name.replace(
            augment_folder.augment_type[0], augment_folder.augment_type[1]
        )
        with open(os.path.join(args.data_dir, augment_folder.difficulty_name + ".txt"), "a") as file:
           file.write(sequence_name + "\n")

        original_files = glob.glob(augment_folder.folder_name + "\*.png")
        for original_file in original_files:
            augmented_file = original_file.replace(
                augment_folder.augment_type[0], augment_folder.augment_type[1]
            )
            augment_folder.augment_type[2](original_file, augmented_file)


if __name__ == '__main__':
    main()
