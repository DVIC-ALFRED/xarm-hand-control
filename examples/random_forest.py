import json
import cv2

import xarm_hand_control.processing.process as xhcpp
from xarm_hand_control.processing.classifiers.random_forest import RandomForest

VIDEO_PATH="/dev/video0"
MODEL_PATH="./models/random_forest.joblib"
DATASET_PATH="./classes.json"


def main():
    with open(DATASET_PATH, "r") as f:
        dataset = json.load(f)

    classes = dataset["classes"]

    classifier = RandomForest()
    classifier.load_model(MODEL_PATH, classes)

    cap = cv2.VideoCapture(VIDEO_PATH)


    xhcpp.process(
        cap,
        classifier=classifier,
        max_num_hands=2
    )


if __name__ == "__main__":
    main()
