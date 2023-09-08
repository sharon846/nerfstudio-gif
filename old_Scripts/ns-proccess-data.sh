#!/bin/bash

# Default values
images="NerfStudioData/Data/Soldier/Images"
output_dir="NerfStudioData/Data/Soldier/Model2"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    key="$1"

    case $key in
        --data)
            images="$2"
            shift
            shift
            ;;
        --output-dir)
            output_dir="$2"
            shift
            shift
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

rm -rf "$output_dir"
mkdir -p "$output_dir"
mkdir -p "$output_dir/colmap"
mkdir -p "$output_dir/colmap/sparse"

colmap feature_extractor --database_path "$output_dir/colmap/database.db" --image_path "$images" --ImageReader.single_camera 1 \
--SiftExtraction.use_gpu 1

colmap exhaustive_matcher --database_path "$output_dir/colmap/database.db" --SiftMatching.use_gpu 1 \
--SiftMatching.min_num_inliers 10 --SiftMatching.max_distance 1.00 \
--SiftMatching.guided_matching 1 --SiftMatching.multiple_models 0

colmap mapper --database_path "$output_dir/colmap/database.db" --image_path "$images" \
--output_path "$output_dir/colmap/sparse" --Mapper.multiple_models 0 \
--Mapper.min_num_matches 10 --Mapper.tri_min_angle 0.5 \
--Mapper.tri_ignore_two_view_tracks 0

python convert.py --output_dir "$output_dir/"
