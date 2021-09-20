import cv2
import xarm_hand_control.processing.process as xhcpp

VIDEO_PATH = "/dev/video0"


def main():
    cap = cv2.VideoCapture(VIDEO_PATH)

    xhcpp.process(cap)


if __name__ == "__main__":
    main()
