import cv2
import os
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup

VIDEO_PATH = "./videoBank/camera0/"
DATAFILE_PATH = "./overview/"
FRAME_PATH = "./png-segments/"
FRAME_WIDTH = 112
FRAME_HEIGHT = 112

with open(DATAFILE_PATH + "train-list.html") as train_file:
    soup = BeautifulSoup(train_file.read(), 'html5lib')

# Get all the training video ids 
train_path = []
train_vids = soup.find_all('tr', {'id': 'name'})
for vid in train_vids:
    vid_id = vid.text.strip('Record ID').strip()
    train_path.append(FRAME_PATH + vid_id + '/')

# print(len(train_path))
print(train_path[0])

tensor_list = []
count = 0
for dir in train_path:
    frames = [] 
    for f in sorted(os.listdir(dir)):
        # Read in gray-scale frame images
        frame = cv2.imread(dir + f, cv2.IMREAD_GRAYSCALE)
        # Resize frame
        resized_frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))
        frames.append(resized_frame.reshape(FRAME_WIDTH, FRAME_HEIGHT, -1))
    # Stack frames into a 3D tensor
    tensor = np.stack(frames, axis=-1)
    tensor_list.append(tensor)

# Normalize data 

# Pad tensors based on max number of frames 
max_num_frames = 140 
tensors = None
for tensor in tensor_list:
    padded_tensor = np.zeros((FRAME_WIDTH, FRAME_HEIGHT, 1, max_num_frames))
    num_frames = tensor.shape[3]
    if num_frames == max_num_frames:
        padded_tensor[:,:,:,:] = tensor[:,:,:,:max_num_frames]
    else:
        padded_tensor[:,:,:,:num_frames] = tensor[:,:,:,:]
    if tensors is None:
        tensors = padded_tensor
    else:
        tensors = np.concatenate([tensors, padded_tensor])

np.save('./data/train_img.npy', tensors)

