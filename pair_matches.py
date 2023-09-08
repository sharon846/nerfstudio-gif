import sys
import sqlite3
import numpy as np
import json
from math import atan2, pi

def image_ids_to_pair_id(image_id1, image_id2):
    if image_id1 > image_id2:
        return 2147483647 * image_id2 + image_id1
    else:
        return 2147483647 * image_id1 + image_id2
    

def pair_id_to_image_ids(pair_id):
    image_id2 = pair_id % 2147483647
    image_id1 = (pair_id - image_id2) / 2147483647
    return int(image_id1), int(image_id2)


#https://towardsdatascience.com/what-are-intrinsic-and-extrinsic-camera-parameters-in-computer-vision-7071b72fb8ec#:~:text=The%20extrinsic%20matrix%20is%20a,to%20the%20pixel%20coordinate%20system.
#https://ww2.mathworks.cn/help/phased/ug/spherical-coordinates.html
#http://motion.cs.illinois.edu/RoboticSystems/3DRotations.html
#https://community.khronos.org/t/get-direction-from-transformation-matrix-or-quat/65502/2

def determine_angel(rotation_matrix):
    x_axis = rotation_matrix[:, 0]
    y_axis = rotation_matrix[:, 1]
    z_axis = rotation_matrix[:, 2]
    
    alpha = atan2(rotation_matrix[2,1], rotation_matrix[2,2])
    beta = np.arcsin(rotation_matrix[2,0])
    gamma = atan2(rotation_matrix[1,0], rotation_matrix[0,0])

    return alpha, beta, gamma

def get_matching_image(colmap_dir, anchor_frame_is_last_frame = True):
    # Read the JSON file
    with open(f'{colmap_dir}/transforms.json', 'r') as f:
        data = json.load(f)

    frames = []
    for frame in data['frames']:
        transform_matrix = frame['transform_matrix']
        transform_matrix = np.array(transform_matrix)
        pos = transform_matrix[:-1,-1]
        alpha, beta, gamma = determine_angel(transform_matrix[:3,:3])

        frames.append([*pos, alpha, beta, gamma])

    frames = np.array(frames)

    # Connect to the database
    conn = sqlite3.connect(f'{colmap_dir}/colmap/database.db')

    # Create a cursor
    cursor = conn.cursor()

    # getting image_id for number of images
    cursor.execute(f"SELECT image_id FROM images")
    rows = cursor.fetchall()
    num_images = len(rows)

    image_matches = []
    start_id = 1
    last_id = num_images

    if anchor_frame_is_last_frame:
        first_image_id = num_images
    else:
        start_id = 2
        last_id = num_images + 1
        first_image_id = 1


    # getting every pair of images the the first image_id is the last image's id
    for second_image_id in range(start_id, last_id):
        # Execute a query
        cursor.execute(f"SELECT pair_id, rows,cols, data FROM matches where pair_id = {image_ids_to_pair_id(first_image_id, second_image_id)}")
        rows = cursor.fetchall()
        image_matches.append(rows)


    # putting data in dictionary
    image_matches_with_image_ids = []
    for row in image_matches:
        try:
            if anchor_frame_is_last_frame:
                image1_id, image2_id = pair_id_to_image_ids(row[0][0])
            else:
                image2_id, image1_id = pair_id_to_image_ids(row[0][0])
            image_matches_with_image_ids.append({"pair_id" : row[0][0],"matches" : row[0][1], "image1_id" : image1_id, "image2_id" : image2_id,  "image1_features": frames[image1_id - 1], "image2_features": frames[image2_id - 1]})
        except Exception as e:
            print(e)

    # sorting data
    sorted_data = sorted(image_matches_with_image_ids, key=lambda item: item['matches'], reverse=True)

    if anchor_frame_is_last_frame:
        id_threshold = (num_images) * (2/3) # we want an image id that is smaller than this
        result = next((item for item in sorted_data if item['image1_id'] < id_threshold), None)
    else:
        id_threshold = (num_images) * (1/3) # we want an image id that is smaller than this
        result = next((item for item in sorted_data if item['image1_id'] > id_threshold), None)

    # for item in sorted_data:
    #     print(item)

    # Close cursor and connection
    cursor.close()
    conn.close()

    return result, sorted_data, frames



# result_end, sorted_data_end, _ = get_matching_image(True)
# result_start, sorted_data_start, _ = get_matching_image(False)

# distance_end = np.linalg.norm(result_end["image1_features"][:3] - result_end["image2_features"][:3]) ** 2 
# distance_start = np.linalg.norm(result_start["image1_features"][:3] - result_start["image2_features"][:3]) ** 2

# start_wins = distance_start < distance_end

# if distance_start < distance_end:
    # print("Start wins... Fatlity")
# else:
    # print("End wins... Flawless Victory")

# print(_.shape)

