import json
import os
from collections import deque
from statistics import mode
from typing import Any, Callable, Iterable, List, Literal, NamedTuple, Tuple, Union
from queue import Queue

import cv2
import mediapipe as mp
import numpy as np

from xarm_hand_control.processing.classifier_base import Classifier
from xarm_hand_control.utils import FPS, ClassificationMode, Command

# * -------------------------------------------------------------------------
# * PROGRAM PARAMETERS
# * -------------------------------------------------------------------------
ROBOT_COMMAND_SCALE = 100
ROBOT_SPEED = 100.0
ROBOT_MVACC = 1000.0
# * -------------------------------------------------------------------------

WINDOW_NAME = "Hand Control"

classification_buffer = deque(maxlen=5)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
drawing_styles = mp.solutions.drawing_styles


def run_inference(
    classifier: Classifier,
    landmarks: Iterable,
) -> list:

    f_data = classifier.format_data(data=landmarks)
    classified_data = classifier.run_classification(f_data)

    # add to buffer and return most common occurence in last n frames
    classification_buffer.appendleft(
        tuple([item["class_name"] for item in classified_data])
    )

    return mode(classification_buffer)


def run_hands(image: Any, hands: mp_hands.Hands) -> Tuple[Any, list]:

    # Convert the BGR image to RGB, flip the image around y-axis for correct
    # handedness output and process it with MediaPipe Hands.
    results = hands.process(cv2.flip(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), 1))

    if not results.multi_hand_landmarks:
        return None, None

    annotated_image = cv2.flip(image.copy(), 1)

    for hand_landmarks in results.multi_hand_landmarks:
        mp_drawing.draw_landmarks(
            annotated_image,
            hand_landmarks,
            mp_hands.HAND_CONNECTIONS,
            drawing_styles.get_default_hand_landmark_style(),
            drawing_styles.get_default_hand_connection_style(),
        )

    return annotated_image, results.multi_hand_landmarks


def get_center_coords(landmarks: list) -> List[float]:

    # palm center as the point between wrist and index metacarpal head
    palm_centers = []
    for landmark in landmarks:
        p1 = (
            landmark.landmark[mp_hands.HandLandmark.WRIST].x,
            landmark.landmark[mp_hands.HandLandmark.WRIST].y,
        )
        p2 = (
            landmark.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP].x,
            landmark.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP].y,
        )

        palm_center = [(p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2]
        palm_center_centered = [palm_center[0] - 0.5, -(palm_center[1] - 0.5)]
        palm_centers.append(palm_center_centered)

    palm_centers_distances = [
        np.linalg.norm(palm_center, ord=2) for palm_center in palm_centers
    ]
    # get index of row with smallest distance to center (ignore angle)
    min_index = np.argmin(palm_centers_distances, axis=0)
    x, y = palm_centers[min_index]

    return x, y


def run_processing(
    classifier: Classifier, landmarks: Iterable
) -> Tuple[str, List[float]]:

    if landmarks is None:
        return "", None

    if classifier is not None:
        classified_hands = run_inference(classifier, landmarks)
    else:
        classified_hands = None

    center_coords = get_center_coords(landmarks)
    x, y = center_coords

    if classified_hands is None:
        to_show_text = f"{x:.2f}, {y:.2f}"
    else:
        classified_hands = ", ".join(classified_hands)
        to_show_text = " | ".join(
            [
                classified_hands,
                f"{x:.2f}, {y:.2f}",
            ]
        )

    return to_show_text, center_coords


def add_image_info(image, top_left_text, bottom_left_text):

    font = cv2.FONT_HERSHEY_SIMPLEX
    top_left_corner_of_text = (20, 30)
    bottom_left_corner_of_text = (20, image.shape[0] - 30)
    font_scale = 0.8
    white = (255, 255, 255)
    red = (0, 0, 255)
    tickness = 2
    linetype = cv2.LINE_AA

    # show fps
    cv2.putText(
        img=image,
        text=top_left_text,
        org=top_left_corner_of_text,
        fontFace=font,
        fontScale=font_scale,
        color=red,
        thickness=tickness,
        lineType=linetype,
    )

    # show hand info
    cv2.putText(
        img=image,
        text=bottom_left_text,
        org=bottom_left_corner_of_text,
        fontFace=font,
        fontScale=font_scale,
        color=white,
        thickness=tickness,
        lineType=linetype,
    )

    # show dot at center of image
    im_center = (int(image.shape[1] / 2), int(image.shape[0] / 2))
    cv2.circle(img=image, center=im_center, radius=3, color=(0, 0, 255), thickness=3)


def process(
    cap: Any,
    classifier: Classifier = None,
    coords_extracter_func: Callable = None,
    max_num_hands: int = 1,
):

    inner_fps = FPS()
    outer_fps = FPS()

    _ = cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_AUTOSIZE)

    # Check if the webcam is opened correctly
    if not cap.isOpened():
        raise IOError("Cannot open webcam")

    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=max_num_hands,
        min_detection_confidence=0.7,
    )

    try:
        while cap.isOpened():
            cap_ok, frame = cap.read()
            if not cap_ok:
                print("cap not ok")
                continue

            inner_fps.update()

            ret_frame, landmarks = run_hands(frame, hands)
            to_show_text, center_coords = run_processing(classifier, landmarks)

            if coords_extracter_func is not None:
                coords_extracter_func(center_coords)

            to_show = cv2.flip(frame, 1) if ret_frame is None else ret_frame

            inner_fps.update()
            outer_fps.update()
            outer_fps_value = int(outer_fps.fps())
            inner_fps_value = int(inner_fps.fps())

            fpss = f"{outer_fps_value}/{inner_fps_value}"

            add_image_info(to_show, fpss, to_show_text)

            cv2.imshow(WINDOW_NAME, to_show)

            key = cv2.waitKey(1)
            if key == 27:
                raise KeyboardInterrupt

    except KeyboardInterrupt:
        cap.release()
        cv2.destroyAllWindows()
        hands.close()
