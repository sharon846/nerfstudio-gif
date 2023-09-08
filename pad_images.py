import argparse
import json
import cv2
import numpy as np
import os

#--output_dir "render" --camera_path "/root/Sharon/camera_path.json"

def pad_image(image_path, is_old):
    image = cv2.imread(image_path)

    # padding for each dimension (top, bottom, left, right, and channels)
    pad_width = [(6, 6), (6, 6), (0, 0)]
    if is_old:
        color = np.array([255, 0, 0], dtype=np.uint8)
    else:
        color = np.array([0, 255, 0], dtype=np.uint8)

    # Create a image of the same size as the padded image filled with the choosen color
    new_im = np.full((image.shape[0] + sum(pad_width[0]), image.shape[1] + sum(pad_width[1]), 3), color, dtype=np.uint8)

    # Copy the original image onto the green padding
    new_im[pad_width[0][0]:-pad_width[0][1], pad_width[1][0]:-pad_width[1][1], :] = image

    cv2.imwrite(image_path, new_im)

# Create an argument parser
parser = argparse.ArgumentParser(description='Run a shell command and capture its output.')
parser.add_argument('--output_dir', type=str, help='Path to the rendered data', required=True)
parser.add_argument('--camera_path', type=str, help='Path to the camera generated data', required=True)

# Parse the command-line arguments
args = parser.parse_args()

with open(args.camera_path, 'r') as f:
    data = json.load(f)

green_start_idx = np.array(data['new_frames'])[0]
#print(green_start_idx)

for idx, img in enumerate(os.listdir(args.output_dir)):
    #print(idx, idx < green_start_idx)
    pad_image(os.path.join(args.output_dir, img), idx < green_start_idx)
