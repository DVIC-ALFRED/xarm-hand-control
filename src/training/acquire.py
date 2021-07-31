import json
import sys

import cv2
import mediapipe as mp

WINDOW_NAME = "win"

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
drawing_styles = mp.solutions.drawing_styles

classes = {0: 'open_hand', 1: 'fist', 2: 'two', 3: 'three', 4: 'spiderman', 5: 'ok', 6: 'pinch',
           7: 'thumb_left', 8: 'thumb_right', 9: 'thumb_up', 10: 'thumb_down',
           11: 'index_left', 12: 'index_right', 13: 'index_up', 14: 'index_down',
           15: 'middle_left', 16: 'middle_right', 17:  'middle_up', 18: 'middle_down',
           19: 'ring_left', 20:  'ring_right', 21: 'ring_up', 22: 'ring_down',
           23: 'little_left', 24: 'little_right', 25: 'little_up', 26: 'little_down'
           }

classes = [
    {"name": 'open_hand'},
    {"name": 'open_hand'},
    {"name": 'fist'},
    {"name": 'two'},
    {"name": 'three'},
    {"name": 'spiderman'},
    {"name": 'ok'},
    {"name": 'pinch'},
    {"name": 'thumb_left'},
    {"name": 'thumb_right'},
    {"name": 'thumb_up'},
    {"name": 'thumb_down'},
    {"name": 'index_left'},
    {"name": 'index_right'},
    {"name": 'index_up'},
    {"name": 'index_down'},
    {"name": 'middle_left'},
    {"name": 'middle_right'},
    {"name": 'middle_up'},
    {"name": 'middle_down'},
    {"name": 'ring_left'},
    {"name": 'ring_right'},
    {"name": 'ring_up'},
    {"name": 'ring_down'},
    {"name": 'little_left'},
    {"name": 'little_right'},
    {"name": 'little_up'},
    {"name": 'little_down'}
]


def run_one_hand(image, hands):
    # Convert the BGR image to RGB, flip the image around y-axis for correct
    # handedness output and process it with MediaPipe Hands.
    image.flags.writeable = False
    results = hands.process(
        cv2.flip(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), 1))
    image.flags.writeable = True

    if not results.multi_hand_landmarks:
        return None, None

    annotated_image = cv2.flip(image.copy(), 1)

    hand_landmarks = results.multi_hand_landmarks[0]
    # Print index finger tip coordinates.

    mp_drawing.draw_landmarks(
        annotated_image, hand_landmarks, mp_hands.HAND_CONNECTIONS,
        drawing_styles.get_default_hand_landmark_style(),
        drawing_styles.get_default_hand_connection_style())

    return cv2.flip(annotated_image, 1), hand_landmarks


def save_landmarks(data, curr_class_index, landmarks):
    if landmarks is None:
        return

    f_landmarks = [[point.x, point.y] for point in landmarks.landmark]

    data.append([f_landmarks, curr_class_index])


def acquire(output_path, video_index):
    data = []

    win = cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_AUTOSIZE)

    cap = cv2.VideoCapture(video_index)

    # Check if the webcam is opened correctly
    if not cap.isOpened():
        raise IOError("Cannot open webcam")

    cap_ok, frame = cap.read()
    cv2.imshow(WINDOW_NAME, frame)
    cv2.waitKey(100)

    for index, gesture_class in classes.items():
        user_input = input(
            f'press enter to acquire for {gesture_class}, "exit" to exit.')
        if user_input == "exit":
            break

        try:
            # Run MediaPipe Hands.
            with mp_hands.Hands(
                    static_image_mode=False,
                    max_num_hands=1,
                    min_detection_confidence=0.7) as hands:

                while cap.isOpened():
                    cap_ok, frame = cap.read()
                    if not cap_ok:
                        continue
                    ret_frame, ret_landmarks = run_one_hand(frame, hands)

                    to_show = frame if ret_frame is None else ret_frame
                    cv2.imshow(WINDOW_NAME, to_show)
                    cv2.waitKey(100)

                    save_landmarks(data, index, ret_landmarks)

        except KeyboardInterrupt:
            print('done.')
            continue

    data_dict = {
        "data": data,
        "classes": classes
    }

    with open(output_path, 'w') as f:
        json.dump(data_dict, f)

    cap.release()
    cv2.destroyAllWindows()
    sys.exit(0)