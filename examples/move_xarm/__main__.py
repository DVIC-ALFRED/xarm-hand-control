import signal
import sys
import threading
import time
from queue import Queue

import cv2
import numpy as np
import xarm_hand_control.processing.process as xhcpp
from xarm.wrapper import XArmAPI

VIDEO_PATH = "/dev/video4"
ARM_IP = "172.21.72.200"

arm: XArmAPI = None

COMMAND_QUEUE = Queue()


def sigint_handler(sig, frame):
    print("\nSIGINT Captured, terminating")
    if arm is not None:
        arm.set_state(4)
        arm.disconnect()

    sys.exit(0)


signal.signal(signal.SIGINT, sigint_handler)


def robot_start() -> XArmAPI:
    arm = "dummy"

    connected = False
    while not connected:
        try:
            arm = XArmAPI(ARM_IP, do_not_open=True)
            arm.connect()
            connected = True
        except:
            print("arm is not online. trying again in 3 seconds...")
            time.sleep(3)

    arm.set_world_offset([0, 0, 0, 0, 0, 0])
    time.sleep(1)

    arm.clean_error()
    arm.motion_enable(enable=True)
    arm.set_mode(0)
    arm.set_state(state=0)

    ret = arm.set_position(
        0,
        -227.8,
        643.9,
        0,
        -90,
        90,
        radius=-1,
        is_radian=False,
        wait=True,
        speed=100,
        mvacc=10000,
        relative=False,
    )
    time.sleep(1)
    print("arm started")

    return arm

def worker(arm):
    SKIPPED_COMMANDS = 10
    COEFF = 22
    counter = 0

    goal_pos = arm.position

    while True:
        item = COMMAND_QUEUE.get()
        counter += 1
        if item is not None and counter > SKIPPED_COMMANDS:
            # print(f'Working on {item}')

            x = item[0] * COEFF
            z = item[1] * COEFF

            goal_pos[0] += x
            goal_pos[2] += z

            speed = np.linalg.norm(item, ord=2) * COEFF * 5
            # speed = np.log(speed) * COEFF
            mvacc = speed * 10
            ret = arm.set_position(
                *goal_pos, speed=speed, mvacc=mvacc, wait=False, relative=False
            )
            if ret < 0:
                print("error")

            counter = 0

        COMMAND_QUEUE.task_done()


def post_command(data):
    COMMAND_QUEUE.put(data)


def main():
    arm = robot_start()
    # arm = ""
    threading.Thread(
        target=worker,
        args=[
            arm,
        ],
        daemon=True,
    ).start()

    cap = cv2.VideoCapture(VIDEO_PATH)

    xhcpp.loop(
        cap,
        coords_extracter_func=post_command
    )

    COMMAND_QUEUE.join()


if __name__ == "__main__":
    main()
