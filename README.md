# Overview
This repository provides functionality to inpaint specific objects in video using a YOLOv9 dataset obtained from Roboflow Universe.

|![sample](sample.gif "sample")|
Ex. Inpaint Number plate

# Requirements
This repository requires the following tools:
- windows
- CUDA_11.8
- python3.10.X
- ffmpeg

It may work with other versions, but it is not guaranteed.

# installation
## Install ffmpeg
This repository uses ffmpeg command.  
Please install from the official FFmpeg website.  

## Install cuda11.8
### Chack NVIDIA Driver

Execute following command and confirm CUDA version is 11.8 or higher.
```
$ nvidia-smi
```

### Check CUDA Toolkit
Execute following command and confirm CUDA version is 11.8 or higher.  
Verify that CUDA_11.8 is built.
```
$ nvcc -V
```

## Create venv
Create venv and pip install.
Execute following command.  
```
$ python -m venv venv
$ venv\Scripts\Activate
$ pip install -r requirements.txt
```

# Quick Start

## Install .pt file
Execute following command in WSL.  
```
$ mkdir weights
$ wget -P weights -q https://github.com/WongKinYiu/yolov9/releases/download/v0.1/gelan-c.pt
```

## Download dataset
Download any yolov9 dataset from the following URL.
https://universe.roboflow.com/


## Train
Input the paths for the downloaded `data.yaml` with `--data` and execute following command.  
"Ensure that the paths specified for 'test', 'train', and 'val' in the data.yaml file are correct.  
For other options, please refer to the contents of `train.py`.
```
$ python train.py --batch 16 --epochs 10 --img 640 --device 0 --min-items 0 --close-mosaic 15 --data {YOUR_DATASET} --weights weights/gelan-c.pt --cfg models/detect/gelan-c.yaml --hyp hyp.scratch-high.yaml
```

## Detect
Inference will be performed using the best.pt obtained from train.py
Input the path of the video you want to run inference on with `--source`.
For other options, please refer to the contents of `detect.py`.
```
$ python detect.py --conf 0.05 --img 1280 --device 0 --weights {TRAIN DATA} --source {VIDEO} --save-txt --save-conf
```

## Inpaint
Edit the following parts of main.py:
- video_path: Path to the original video (the one specified with --source in detect.py)
- output_video_path: Path to the final output video
- labelsdir: Path to the labels directory (results obtained from detect.py)
- file_prefix: Filename prefix of the labels in the directory (the part before _number)
- fps: Frame rate of the original video"

```
$ python inpaint.py
```

# Execute the detect and inpaint workflow at all once
Edit main.py and input following variables.
WEIGHTS_PATH: Your .pt file
VIDEO_PATH: Video path that you want to inpaint
OUTPUT_PATH: Output video path you want to

```
$ python main.py
```