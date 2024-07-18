#!/bin/bash

WEIGHTS_PATH="$1"
VIDEO_PATH="$2"
OUTPUT="$3"

EXTENSION=$(echo "$VIDEO_PATH" | awk -F. '{print $NF}')
BASE_NAME=$(basename "$VIDEO_PATH" ".$EXTENSION")

source venv/Scripts/activate

python detect.py --conf 0.05 --img 1280 --device 0 --name "$BASE_NAME" --weights "$WEIGHTS_PATH" --source "$VIDEO_PATH" --save-txt --save-conf

python inpaint.py --inputvideo "$VIDEO_PATH" --labelsdir "runs/detect/$BASE_NAME/labels" --outputvideo $OUTPUT