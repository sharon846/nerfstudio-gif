#!/bin/bash

# Define variables from command line arguments
scripts_dir=$(dirname "$(readlink -f "$0")")
data_dir="/root/Sharon/Data/$1"

output_dir="$data_dir/outputs/colmap/nerfacto"
subdirs=($(find "$output_dir" -maxdepth 1 -type d -printf "%T@ %p\n" | sort -n | awk '{print $2}'))

model_dir="${subdirs[-1]}"
colmap_dir="$data_dir/colmap"

cd $data_dir && rm -rf render plots combined && mkdir plots combined

python "$scripts_dir/generate_spline.py" --colmap_dir "$colmap_dir" --dataparser_dir "$model_dir"
ffmpeg -y -framerate 30 -i plots/plot%05d.png -c:v libx264 -r 30 -pix_fmt yuv420p outputs/camera.mp4

ns-render camera-path --load-config "$model_dir/config.yml" --camera-path-filename "$model_dir/camera_path.json" \
--output-format images --output-path "render"
ffmpeg -y -framerate 30 -i render/%05d.jpg -c:v libx264 -r 30 -pix_fmt yuv420p outputs/video.mp4

echo "editing final render"
python "$scripts_dir/pad_images.py" --output_dir "render" --camera_path "$model_dir/camera_path.json"

python "$scripts_dir/combine_video_camera.py" --data_dir "$data_dir"
ffmpeg -y -framerate 30 -i combined/%05d.png -c:v libx264 -r 30 -pix_fmt yuv420p outputs/cam_and_vid.mp4

#echo $model_dir
#echo $colmap_dir
