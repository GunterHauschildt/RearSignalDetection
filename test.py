from time import strftime, gmtime, localtime
from typing import Any
import tensorflow as tf

from model.cnn_lstm import SimpleCNNLSTM
from tf_records import parse_tfrecord
import cv2
import numpy as np
import os
import glob
import random
import argparse
from utils import *
from collections import defaultdict


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights-file-name', type=str, required=True)
    parser.add_argument('--data-dir', type=str, default=None)
    parser.add_argument('--root-dir', type=str, default=None)
    
    args = parser.parse_args()
    
    if args.root_dir is None or args.root_dir is not None and not os.path.isdir(args.root_dir):
        print(f"Can't open {args.root_dir}.")
        exit(-1)

    if args.weights_file_name is None:
        print(f"Don't have a weights file name. Quitting.")
        exit(-1)

    root_dir = None
    if os.path.isdir("c:"):
        root_dir = "C:\\Users\\gunte\\OneDrive\\Desktop\\Projects\\BrakeLightDetection"
    elif os.path.isdir("/mnt/c"):
        root_dir = "/mnt/c/Users/gunte/OneDrive/Desktop/Projects/BrakeLightDetection"

    if root_dir is None or not os.path.isdir(root_dir):
        print(f"Unable to find {root_dir}.")
        exit(-1)

    shape = (32, 128, 128, 3)
    num_units = 1024
    num_classes = max(SequenceTypes.types, key=lambda seq: seq.classification).classification + 1
    nn = SimpleCNNLSTM(shape, num_units, num_classes)
    nn.model().summary()

    model_dir = os.path.join(root_dir, "data", "trained_models")

    weights_file_name = os.path.join(model_dir, "checkpoints", args.weights_file_name)
    if os.path.isfile(weights_file_name + ".index"):
        try:
            nn.model().load_weights(weights_file_name)
            print(f"Loaded weights file: {weights_file_name}")
        except:
            print(f"Error loading weights file: {weights_file_name}. Quitting.")
            exit(-1)
    else:
        print(f"No weights file: {weights_file_name}. Quitting.")
        exit(-1)

    sequence_folders = []
    ordered_sequences = [
        (("OLR", "BLR"), "Hazard"),
        (("OLO", "BLO"), "Left Turn"),
        (("OOR", "BOR"), "Right Turn"),
        (("OOO", "OOO"), "No Signal")
    ]
    folder_names = []
    for folder_name in os.walk(args.data_dir):
        folder_name = folder_name[0]
        if folder_name.endswith("light_mask"):
            folder_names.append(folder_name)
    random.shuffle(folder_names)
    for folder_name in folder_names:
        for sequence in ordered_sequences:
            if sequence[0][0] in folder_name or sequence[0][1] in folder_name:
                if len(glob.glob(os.path.join(folder_name, "*.png"))) > 60:
                    sequence_folders.append(folder_name)

    random.shuffle(sequence_folders)

    results = defaultdict(lambda: int(0))
    for sequence_folder in sequence_folders:

        sequence_numpy = image_sequence_folder_to_numpy(
            sequence_folder,
            shape[0],
            shape[1:3]
        )

        sequence_numpy = (sequence_numpy[:, :, :, ::-1] / 255.).astype(np.float32)
        sequence_numpy = np.expand_dims(sequence_numpy, axis=0)
        p = nn.model().predict(sequence_numpy)
        p = np.argmax(p, axis=1)[0]
        pred_name = f"{ClassificationNames[p]}"
        results[pred_name] += 1
        if results[pred_name] > 10:
            continue

        sequence_list = image_sequence_folder_to_list(
            sequence_folder
        )
        sequence_list = sequence_list[:150]

        w = 0
        h = 0
        for s in range(len(sequence_list)):
            image = sequence_list[s]
            w = max(w, image.shape[1])
            h = max(h, image.shape[0])

        folder = "C:\\Users\\gunte\\OneDrive\\Desktop\\Projects\\BrakeLightDetection\\data\\videos"
        name = str(results[pred_name]) + ".mp4"
        path = os.path.join(folder, pred_name + "_" + name)
        video_writer = cv2.VideoWriter(
            path,
            cv.VideoWriter.fourcc(*'mp4v'),
            33,
            (w, h)
        )

        for s in range(len(sequence_list)):
            image = np.zeros((h, w, 3)).astype(np.uint8)
            m = sequence_list[s]

            image[0:m.shape[0], 0:m.shape[1]] = m

            # image = cv.putText(image, f"{ClassificationNames[p]}", (20, 20),
            #                    cv.FONT_HERSHEY_PLAIN, 1.0, (255, 255, 255))
            cv.imshow("pred", image)
            cv.waitKey(1)
            video_writer.write(image)



if __name__ == '__main__':
    main()
