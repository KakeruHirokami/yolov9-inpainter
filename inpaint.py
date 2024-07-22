import cv2
import os
import subprocess
import argparse
import shutil

import numpy as np

def import_labels(labelsdir, file_prefix, movemean_maxframe):
   """
   Add two numbers and return the result.

   Parameters:
   labelsdir (str): Your dataset's label directory path.
   file_prefix (str): The first part of the label file name (the string before '_XX.txt' in the label file name).
   movemean_maxframe (int): Maximum number of frame to calculate the moving average if no instance is detected.

   Returns:
   int or float: The sum of the two numbers.
   """
   cropnum = count_files_in_directory(labelsdir)   # Get number of files
   bboxlist_time_series = []
   for i in range(cropnum):
      frameno = i + 1
      filename = f"{labelsdir}/{file_prefix}_{frameno}.txt"
      if os.path.exists(filename):
         print(f"'{filename}' was found.")
         with open(filename) as f:
            detectlist = []
            labels = f.readlines()
            for label in labels:
               detectlist.append(list(map(float, label.split(" "))))
            bboxlist_time_series.append(detectlist)
      else:
         print(f"'{filename}' was not found.")
         search_frame_num = int(movemean_maxframe/2)
         # Search backward in time
         st_nums = []
         st_files = []
         for j in range(search_frame_num):
            frame_num = frameno-j-1
            filename = f"{labelsdir}/{file_prefix}_{frame_num}.txt"
            if os.path.exists(filename):
               st_nums.append(frame_num)
               st_files.append(filename)
         # Search advances in time
         ed_nums = []
         ed_files = []
         for j in range(search_frame_num):
            frame_num = frameno+j+1
            filename = f"{labelsdir}/{file_prefix}_{frame_num}.txt"
            if os.path.exists(filename):
               ed_nums.append(frame_num)
               ed_files.append(filename)

         if st_nums != [] and ed_nums != [] and (ed_nums[0] - st_nums[0] <= int(movemean_maxframe)):
            # If not detect period less than movemean_maxframe
            
            # If st_file or ed_file detects two or more instances, detect the corresponding instance because it is unknown which instance it corresponds to
            # The one that is closer is assumed to be the same instance
            # If the same instance does not exist, "None" is returned.
            max_threshold = 0.2   # Maximum amount of movement considered as the same instance (percentage)
            for st_num, st_file in zip(st_nums, st_files):
               with open(st_file) as f:
                  st_labels = f.readlines()
                  st_labels = list(map(lambda k: k.split(" "), st_labels))
               for ed_num, ed_file in zip(ed_nums, ed_files):
                  with open(ed_file) as f:
                     ed_labels = f.readlines()
                     ed_labels = list(map(lambda k: k.split(" "), ed_labels))
                  match_list = matching(st_labels, ed_labels, max_threshold)
                  if match_list:
                     break
               if match_list:
                  break
            if match_list:
               no = frameno - st_nums[0] # Number of frames from st_num
               movemean_list_list = []
               for match_tuple in match_list:
                  st_label_array = np.array(list(map(float, st_labels[match_tuple[0]][1:5])))
                  ed_label_array = np.array(list(map(float, ed_labels[match_tuple[1]][1:5])))
                  st_diff_array = (ed_label_array - st_label_array) * no / (ed_num - st_nums[0])
                  movemean_list_list.append([0] + list(st_label_array + st_diff_array) + [0.5])
               bboxlist_time_series.append(movemean_list_list)
            else:
               # If there is not match_list, no detection is assumed.
               bboxlist_time_series.append([[]])
         else:
            # If there is not match_list, no detection is assumed.
            print(f'not detected label. frameno: {frameno}')
            bboxlist_time_series.append([[]])

   return bboxlist_time_series

def matching(st_labels, ed_labels, max_threshold):
   match_list = []
   for st_idx, st_label in enumerate(st_labels):
      diff_list = []
      for ed_idx, ed_label in enumerate(ed_labels):
         st_array = np.array(list(map(float,st_label[1:5])))
         ed_array = np.array(list(map(float,ed_label[1:5])))
         diff_array = np.abs(st_array - ed_array)
         diff_list.append(diff_array.sum())
      if max_threshold < min(diff_list):
         # If The least mobile instances exceed max_threshold, recognize that the same instance does not exist
         return None
      min_diff_idx = np.argmin(diff_list)
      match_list.append((st_idx, min_diff_idx))

   # (st_index, ed_index)
   return match_list
         
# get codec
def get_codec(cap):
   fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
   codec = "".join([chr((fourcc >> 8 * i) & 0xFF) for i in range(4)])
   return codec


# Change from video to frame images
def extract_frames(video_path, frames_folder, output_video_path, fps, bboxlist_time_series):
   cap = cv2.VideoCapture(video_path)
   codec = get_codec(cap)
   print(f"input video codec: {codec}")
   print(f"output video codec: h264")
   #fourcc = cv2.VideoWriter_fourcc('h', '2', '6', '4')
   fourcc = cv2.VideoWriter_fourcc(*'mp4v')
   ret, frame = cap.read()
   height, width, ch = frame.shape
   print(frame.shape)
   video = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

   if not cap.isOpened():
      print("Error: Could not open video.")
      return

   if not os.path.exists(frames_folder):
      os.makedirs(frames_folder)

   frame_index = 0
   bbox_color = (0, 255, 0)
   bbox_thickness = 2
   max_area = 0.2 # Maximum area of bbox (to prevent false detection of bboxes that are too big)
   while True:
      print(f"frame_index: {frame_index}")
      if frame_index < len(bboxlist_time_series):
         bboxlist = bboxlist_time_series[frame_index]
         for bbox in bboxlist:
            if bbox == []:
               break
            x_center, y_center, w, h = bbox[1:5]
            if w * h > max_area:
               continue
            w = int(w * width * 0.8)
            h = int(h * height * 0.8)
            x = int(x_center * width) - w // 2
            y = int(y_center * height) - h // 2
            # Create mask iamge
            mask = np.zeros(frame.shape[:2], dtype=np.uint8)
            mask[y:y+h, x:x+w] = 255
            frame = cv2.inpaint(frame, mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)
            #cv2.rectangle(frame, (x, y), (x + w, y + h), bbox_color, bbox_thickness)
      cv2.imwrite(f"{frames_folder}/frame_{frame_index:04d}.png", frame)
      video.write(frame)
      frame_index += 1
      ret, frame = cap.read()
      if not ret:
         break

   video.release()

   cap.release()
   print(f"Extracted {frame_index} frames.")


def count_files_in_directory(directory):
    # List all files in the directory
    all_items = os.listdir(directory)
    # Count files
    file_count = max(int(item.split("_")[-1].split(".")[0]) for item in all_items if os.path.isfile(os.path.join(directory, item)))
    return file_count

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--inputvideo', type=str, help='input video path')
    parser.add_argument('--labelsdir', type=str, help='label directory path')
    parser.add_argument('--outputvideo', type=str, help='output video path')
    opt = parser.parse_args()
    return opt

def get_video_fps(video_path):
   cap = cv2.VideoCapture(video_path)
   if not cap.isOpened():
      print(f"Error: Could not open video {video_path}")
      return None
   fps = cap.get(cv2.CAP_PROP_FPS)
   cap.release()
   return fps

if __name__ == "__main__":
   opt = parse_opt()

   video_path = opt.inputvideo
   labelsdir = opt.labelsdir
   video_path_name = video_path.split("/")[-1].split(".")[0]
   #output_video_path = f'{video_path_name}.mp4'
   output_video_path = opt.outputvideo
   file_prefix = video_path.split("/")[-1].split(".")[0]
   fps = round(get_video_fps(video_path))

   frames_folder = 'frames'
   movemean_maxtime = 2 # Maximum number of seconds to calculate the moving average if no instance is detected.
   
   # Load labels and convert array
   # return
   #  bboxlist_time_series[frameno][instanceno][section]
   #  section=0: classNo
   #  section=1: horizontal center
   #  section=2: vertical center 
   #  section=3: width
   #  section=4: height
   #  section=5: confidence
   movemean_maxframe = movemean_maxtime * fps
   bboxlist_time_series = import_labels(labelsdir, file_prefix, movemean_maxframe)

   # Convert video to images and edit images
   output_ext = output_video_path.split(".")[-1]
   tmp_output_video_path = f"tmp.{output_ext}"
   extract_frames(video_path, frames_folder, tmp_output_video_path, fps, bboxlist_time_series)

   ## Use FFmpeg to extract the audio from the original video and merge it back together
   subprocess.call([
      'ffmpeg',
      '-i', tmp_output_video_path,
      '-i', video_path,
      '-c:v', 'copy',
      '-c:a', 'aac',
      '-strict', 'experimental',
      '-map', '0:v:0?',
      '-map', '1:a:0?',
      output_video_path,
      '-y'
   ])
   os.remove(tmp_output_video_path)
   shutil.rmtree(frames_folder)