import cv2
import numpy as np
import os
import argparse

parser = argparse.ArgumentParser(description='Run a shell command and capture its output.')
parser.add_argument('--data_dir', type=str, help='Path to the rendered data', required=True)

# Parse the command-line arguments
args = parser.parse_args()

num_frames = len(os.listdir(f'{args.data_dir}/render'))
frames = np.array([cv2.imread(f'{args.data_dir}/render/{i:05d}.jpg') for i in range(num_frames)])
cameras = np.array([cv2.imread(f'{args.data_dir}/plots/plot{i:05d}.png') for i in range(num_frames)])

h = frames[0].shape[0]
scaling_factor = h / cameras[0].shape[0]
camera_width = int(cameras[0].shape[1] * scaling_factor)

w = frames[0].shape[1] + camera_width
rw = int(w/2) * 2 + 2

new_images = np.zeros((num_frames, h, rw, 3))
new_images[:,:,1:1+frames[0].shape[1]] = frames
new_images[:,:,1+frames[0].shape[1]:] = np.array([cv2.resize(img, (camera_width, h)) for img in cameras])

for i, img in enumerate(new_images):
        cv2.imwrite(f'{args.data_dir}/combined/{i:05d}.png', img)
