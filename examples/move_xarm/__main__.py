import signal
import threading
import time
from queue import Queue
import sys

import xarm_hand_control as xhc
from xarm.wrapper import XArmAPI
from xarm_hand_control.modules.utils import ClassificationMode

VIDEO_INDEX=4
ARM_IP = "172.21.72.200"

arm = None

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

    arm.motion_enable(enable=True)
    arm.set_mode(0)
    arm.set_state(state=0)

    ret = arm.set_position(0, -227.8, 643.9, 0, -90, 90,
        radius=-1, is_radian=False, wait=True, speed=100, mvacc=10000, relative=False)

    time.sleep(1)
    print("arm started")

    return arm

def worker(arm: XArmAPI):
    counter = 0
    while True:
        item = COMMAND_QUEUE.get()
        counter += 1
        if item is not None and counter > 7:
            print(f'Working on {item}')

            COEFF=30
            x = item[0] * COEFF
            z = item[1] * COEFF
            ret = arm.set_position(x, 0, z, 0, 0, 0, wait=False, relative=True)
            if ret < 0:
                print("error")

            counter = 0

        COMMAND_QUEUE.task_done()


def main():
    arm = robot_start()
    # arm = ""
    threading.Thread(target=worker, args=[arm, ], daemon=True).start()

    xhc.process(
        classification_mode=ClassificationMode.NO_CLASSIFICATION,
        video_index=VIDEO_INDEX,
        robot_command_queue=COMMAND_QUEUE,
        dataset_path=None,
        model_path=None
    )

    COMMAND_QUEUE.join()


if __name__ == "__main__":
    main()
