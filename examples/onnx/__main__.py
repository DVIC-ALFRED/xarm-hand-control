import xarm_hand_control as xhc
from xarm_hand_control.modules.utils import ClassificationMode

VIDEO_INDEX=4


def main():
    xhc.process(
        ClassificationMode.RANDOM_FOREST,
        video_index=VIDEO_INDEX,
        robot_command_queue=None,
        dataset_path="/home/alfred/Documents/ALFRED/xarm-hand-control/classes.json",
        model_path="/home/alfred/Documents/ALFRED/xarm-hand-control/models/random_forest.joblib"
    )


if __name__ == "__main__":
    main()
