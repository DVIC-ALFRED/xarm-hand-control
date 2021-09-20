import xarm_hand_control as xhc
from xarm_hand_control.modules.utils import ClassificationMode

VIDEO_INDEX=0


def main():
    xhc.process(
        ClassificationMode.ONNX,
        video_index=VIDEO_INDEX,
        robot_command_queue=None,
        dataset_path="/home/dimitri/Documents/Projects/ALFRED/xarm-hand-control/classes.json",
        model_path="/home/dimitri/Documents/Projects/ALFRED/xarm-hand-control/models/model.onnx"
    )


if __name__ == "__main__":
    main()
