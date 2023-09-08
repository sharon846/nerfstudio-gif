import subprocess
import argparse
import json
import numpy as np
import os
import shutil

# Create an argument parser
parser = argparse.ArgumentParser(description='Run a shell command and capture its output.')
parser.add_argument('--colmap_dir', type=str, help='Path to the COLMAP directory', required=True)
parser.add_argument('--model_dir', type=str, help='Path to the model trained model', required=True)
parser.add_argument('--model_name', type=str, help='model name', required=True)

# Parse the command-line arguments
args = parser.parse_args()

command = f'cd renders/{args.model_name} && for file in *.jpg; do ffmpeg -i "$file" "${{file%.jpg}}.png"; done && rm -rf *.jpg'
# Run the command and capture its output
output = subprocess.check_output(command, shell=True, encoding='utf-8')

# Print the output
print(output)
exit()
with open('camera_path.json', 'r') as f:
		data = json.load(f)

original_frames = np.array(data['origin_frames'])
new_frames = np.array(data['new_frames'])

min = np.min(new_frames)
for idx in new_frames:
  o = idx - min
  os.rename(f'renders/{args.model_name}/{o:05d}.png', f'renders/{args.model_name}/{idx:05d}.png')

min = np.min(original_frames)
for idx in original_frames:
  n = idx - min
  o = idx + 1
  shutil.copy(f'{args.model_name}/images/frame_{o:05d}.png', f'renders/{args.model_name}/{n:05d}.png')
