import subprocess

WEIGHTS_PATH="./runs/train/exp5/weights/best.pt"
VIDEO_PATH="20240630_085506000_iOS.MOV"
OUTPUT_PATH="output.mp4"

subprocess.call([
    "sh",
    "inpaint_workflow.sh",
    WEIGHTS_PATH,
    VIDEO_PATH,
    OUTPUT_PATH
])