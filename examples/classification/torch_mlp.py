import json
import cv2

import xarm_hand_control.processing.process as xhcpp
from xarm_hand_control.processing.classifiers.torch_mlp import TorchMLP

VIDEO_PATH="/dev/video0"
MODEL_PATH="./models/best.pt"
DATASET_PATH="./classes.json"


def main():
    with open(DATASET_PATH, "r") as f:
        dataset = json.load(f)

    classes = dataset["classes"]

    classifier = TorchMLP()
    classifier.load_model(MODEL_PATH, classes)

    cap = cv2.VideoCapture(VIDEO_PATH)


    xhcpp.loop(
        cap,
        classifier=classifier,
        max_num_hands=2
    )


if __name__ == "__main__":
    main()
