import json

colmap_dir = 'Data/Egg/Colmap'

with open(f'{colmap_dir}/transforms.json', 'r') as f:
	data = json.load(f)
	
# Sort the frames based on the "file_path"
sorted_frames = sorted(data["frames"], key=lambda x: x["file_path"])

# Update the original data with sorted frames
data["frames"] = sorted_frames

# If you want to save the sorted JSON back to a file
with open(f'{colmap_dir}/transforms.json', "w") as json_file:
	json.dump(data, json_file, indent=4)
