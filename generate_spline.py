import pandas as pd
import numpy as np
from scipy.interpolate import splrep, BSpline
from scipy import interpolate
import matplotlib.pyplot as plt
from scipy.spatial.distance import euclidean
import subprocess
import sys
from scipy.integrate import quad, IntegrationWarning
from pair_matches import *
from save_graph import *
import json
import math
import os
import warnings
import argparse

# Suppress specific warnings
warnings.filterwarnings("ignore", category=IntegrationWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

curve_outliers = None

def build_rotation_matrix(angles):

        alpha, beta, gamma = angles

        c,s = np.cos(alpha), np.sin(alpha)
        rot_alpha = np.array([[1,0,0],\
                                                                                                [0,c,-s],\
                                                                                                [0,s,c]])

        c,s = np.cos(beta), np.sin(beta)
        rot_beta = np.array([[c,0,-s],\
                                                                                                [0,1,0],\
                                                                                                [s,0,c]])

        c,s = np.cos(gamma), np.sin(gamma)
        rot_gamma = np.array([[c,-s,0],\
                                                                                                [s,c,0],\
                                                                                                [0,0,1]])

        return rot_gamma @ rot_beta @ rot_alpha

def retrieve_loop_frames():
        start = 1
        end = 0

        result_end, sorted_data_end, _ = get_matching_image(args.colmap_dir, True)
        result_start, sorted_data_start, frms = get_matching_image(args.colmap_dir, False)

        distance_start, distance_end = np.inf, np.inf

        if result_end is not None:
                        distance_end = np.linalg.norm(result_end["image1_features"][:3] - result_end["image2_features"][:3]) ** 2
        if result_start is not None:
                        distance_start = np.linalg.norm(result_start["image1_features"][:3] - result_start["image2_features"][:3]) ** 2

        start_wins = distance_start < distance_end

        if (start_wins):
                        start = int(result_start['image2_id'])
                        end = int(result_start['image1_id'])
        else:
                        start = int(result_end['image1_id'])
                        end = int(result_end['image2_id'])

        loop_detected = np.concatenate((frms[:start-1], frms[end:]), axis=0)
        loop_detected = loop_detected[:,:3]
        return frms[start-1:end], start-1, loop_detected

def derivative_magnitude(t, spline):
        der = interpolate.splev(t, spline, der=1)
        return np.linalg.norm(der)

def generate_spline(data, s=0.5, per=True):
        dims = data.shape[1]
        features = [data[:,d] for d in range(dims)]

        spline, u = interpolate.splprep(features, s = s, k = dims)                        #originally k=3
        new_points = np.array(interpolate.splev(u, spline)).T

        length_curve, _ = quad(derivative_magnitude, 0, 1, args=(spline))
        if per:
                        spline_rep, u_rep = interpolate.splprep(features, s = s * 2, k = dims+1, per=True)
                        points_add = np.array(interpolate.splev(u_rep, spline_rep)).T
                        new_points[:10] = points_add[0:10]
                        new_points[-10:] = points_add[-10:]

        return new_points, [length_curve, u, spline]

def generate_points_on_spline(spline, u_start, u_end, frames_num):
                u = np.linspace(u_start, u_end, frames_num)
                return np.array(interpolate.splev(u, spline)).T

#FOV: https://stackoverflow.com/questions/39992968/how-to-calculate-field-of-view-of-the-camera-from-camera-intrinsic-matrix

def generate_final_csv(frames_idx, added_pos, added_angs, fps=30):
        frames_idx = np.array(frames_idx) + removed_from_start
        with open(f'{args.colmap_dir}/transforms.json', 'r') as f:
                data = json.load(f)
                frames = [frame['transform_matrix'] for frame in data['frames']]
                frames = np.array(frames)
                applied_transform = np.array(data['applied_transform'])
                applied_transform = np.concatenate((applied_transform, np.array([[0,0,0,1]])), axis=0)
                w,h = int(data['w']), int(data['h'])

                fx, fy = float(data['fl_x']), float(data['fl_y'])
                fov = np.rad2deg(2 * np.arctan2(h, 2 * fy))
                #fov = 56.8972479389                    #which is 2 * atan(25.4 / 2.55 (from ios) * sqrt(2) * 1 / 26 (focal length ios)) in degrees

        with open(f'{args.dataparser_dir}/dataparser_transforms.json', 'r') as f:
                data = json.load(f)
                poses_matrix = np.array(data["transform"])
                poses_matrix = np.concatenate((poses_matrix, np.array([[0,0,0,1]])), axis=0)
                poses_matrix = poses_matrix @ np.linalg.inv(applied_transform)
                scale = np.array(data["scale"])

        frames = poses_matrix @ frames
        #frames = np.matmul(poses_matrix, frames)
        frames[:,:3,3] *= scale

        total_frames = len(frames_idx) + len(added_pos)
        video_len = total_frames / fps
        print(f'Total frames: {total_frames}, video_len: {video_len}')
        #video_len = len(added_pos) / fps                                          # disable if doing all cameras, and not just the new ones

        dict =  {'origin_frames': frames_idx.tolist(), 'new_frames': (np.arange(0,len(added_pos)) + len(frames_idx)).tolist(), \
                                'keyframes': [], 'camera_type': 'fisheye', 'render_height': h, 'render_width': w, \
                                'camera_path': [], 'fps': fps, 'seconds': video_len, 'smoothness_value': 2, 'is_cycle': False, 'crop': None}

        u = np.linspace(0, 1, total_frames)                      #change to total_frames if doing all cameras
        ''' enable if want to render the original cameras too '''
        for counter, idx in enumerate(frames_idx):
                keyframe = {'matrix': [], 'fov': fov, 'aspect': 1, 'properties': f'[["FOV",{fov}],["NAME","{counter}"],["TIME",{u[counter]}]]'}
                keyframe['matrix'] = frames[idx]

                arr = keyframe['matrix'].T.flatten()
                keyframe['matrix'] = "[" + ",".join(map(str, arr)) + "]"
                dict['keyframes'].append(keyframe)

                camera_path = {'camera_to_world': [], 'fov': fov, 'aspect': 1}
                camera_path['camera_to_world'] = frames[idx].flatten().tolist()
                dict['camera_path'].append(camera_path)

        for idx in range(len(added_pos)):
                #if idx == 0:
                #          continue
                counter = idx + len(frames_idx)

                Rotation_matrix = build_rotation_matrix(added_angs[idx])

                Camera_matrix = np.zeros((4,4))
                Camera_matrix[:3,:3] = Rotation_matrix
                Camera_matrix[:3,3] = added_pos[idx]
                Camera_matrix[3,3] = 1
                Camera_matrix = poses_matrix @ Camera_matrix
                Camera_matrix[:3,3] *= scale

                keyframe = {'matrix': [], 'fov': fov, 'aspect': 1, 'properties': f'[["FOV",{fov}],["NAME","new {counter}"],["TIME",{u[counter]}]]'}  # enable if doing all cameras
                #keyframe = {'matrix': [], 'fov': fov, 'aspect': 1, 'properties': f'[["FOV",{fov}],["NAME","new {counter}"],["TIME",{u[idx]}]]'}
                keyframe['matrix'] = Camera_matrix.T.flatten()
                keyframe['matrix'] = "[" + ",".join(map(str, keyframe['matrix'])) + "]"
                dict['keyframes'].append(keyframe)

                camera_path = {'camera_to_world': [], 'fov': fov, 'aspect': 1}
                camera_path['camera_to_world'] = Camera_matrix.flatten().tolist()
                dict['camera_path'].append(camera_path)

        # Convert the dictionary to a JSON string
        json_string = json.dumps(dict, indent=4)  # Optional: Use 'indent' for pretty formatting

        # Write the JSON string to a file
        with open(f'{args.dataparser_dir}/camera_path.json', "w") as json_file:
                json_file.write(json_string)


##########                                        GENERATE SPLINE OF POSITIONS                                  ##########

parser = argparse.ArgumentParser(description='Run a shell command and capture its output.')
parser.add_argument('--colmap_dir', type=str, help='Path to the COLMAP directory', required=True)
parser.add_argument('--dataparser_dir', type=str, help='Path to the model trained model', required=True)

# Parse the command-line arguments
args = parser.parse_args()


frames, removed_from_start, loop_detected = retrieve_loop_frames()
num_data = int(len(frames) / 2)

positions = frames[:,:3]
positions_inv = np.concatenate((positions[-num_data:], positions[:num_data]), axis=0)
angles = frames[:,3:]

new_points, _ = generate_spline(data=positions_inv, s=2)


##########                                        DECIDE FINAL FRAMES BY POSITIONS
##########

distances = np.linalg.norm(new_points - positions_inv, axis=1)
distances = np.concatenate((distances[-num_data:], distances[:num_data]), axis=0)
distance_threshold = 3.5 * np.min(distances)

not_on_spline = np.where(distances > distance_threshold)[0]

breaks = np.where(np.diff(not_on_spline) != 1)[0] + 1
sequences = np.split(not_on_spline, breaks)

not_on_spline = []
if sequences[0][0] == 0:
                not_on_spline += list(sequences[0])
if sequences[-1][-1] == len(positions) - 1:
                not_on_spline += list(sequences[-1])

not_on_spline = np.array(not_on_spline)
curve_outliers = positions[not_on_spline]
print(not_on_spline)


##########                                        REMOVE OLD FRAMES                                        ##########

frames = np.arange(0, len(positions), 1)
frames = np.delete(frames, not_on_spline)
num_data = 2
#num_data = int(len(frames) / 2)

positions = np.delete(positions, not_on_spline, axis=0)
_, dt = generate_spline(data=positions, per=False)
curve_len = dt[0]

positions_inv = np.concatenate((positions[-num_data:], positions[:num_data]), axis=0)

angles = np.delete(angles, not_on_spline, axis=0)
angles_inv = np.concatenate((angles[-num_data:], angles[:num_data]), axis=0)

##########                                        ADD NEW FRAMES (POSITIONS)                                      ##########

_, dt = generate_spline(data=positions_inv, per=False)
u, spline = dt[1], dt[2]

u_start, u_end = u[num_data-1], u[num_data]
curve_len_added, _ = quad(derivative_magnitude, u_start, u_end, args=(spline))

num_frames_to_add = curve_len_added * len(frames) / curve_len
num_frames_to_add = int(num_frames_to_add + 2)

generated_points = generate_points_on_spline(spline=spline, u_start=u_start, u_end=u_end, frames_num=num_frames_to_add)
generated_points = generated_points[1:-1]                          #to exclude the first and last one, but maintain the needed frames number

##########                                        ADD NEW FRAMES (ANGLES)                                ##########

new_angles, dt = generate_spline(data=angles_inv, per=False)
u, spline = dt[1], dt[2]

u_start, u_end = u[num_data-1], u[num_data]
generated_angles = generate_points_on_spline(spline=spline, u_start=u_start, u_end=u_end, frames_num=num_frames_to_add)
generated_angles = generated_angles[1:-1]

generate_final_csv(frames, generated_points, generated_angles)

#save html 3d viewer of points
save_viewer(f'{args.colmap_dir}/../outputs', positions, loop_detected, curve_outliers, generated_points)

#save frames in plots, to create camera movment animation
save_camera(f'{args.colmap_dir}/../plots', f'{args.dataparser_dir}/camera_path.json', positions, loop_detected, curve_outliers, generated_points)

exit()

##########                                        PLOTTING                                              ##########

#print(removed_from_start)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.view_init(azim=-90, elev=80, roll=0)

# Original points
ax.scatter(positions[:,0],positions[:,1],positions[:,2],color='green', label='all points')
ax.scatter(generated_points[:,0], generated_points[:,1], generated_points[:,2], color='blue', label='Added points')

if len(loop_detected) > 0:
        ax.scatter(loop_detected[:,0], loop_detected[:,1], loop_detected[:,2], color='black', label='Removed on loop detection')
if len(curve_outliers) > 0:
        ax.scatter(curve_outliers[:,0], curve_outliers[:,1], curve_outliers[:,2], color='red', label='Removed on outliers detection')

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.legend()

fig.savefig(f"{args.colmap_dir}/../outputs/points.png", dpi=300)

fig_p = plt.figure()
ax_p = fig_p.add_subplot(111, projection='3d')

# Original points
ax_p.scatter(angles[:,0], angles[:,1], angles[:,2], color='green', label='all points')
ax_p.scatter(generated_angles[:,0], generated_angles[:,1], generated_angles[:,2], color='blue', label='Smooth Spline')

ax_p.set_xlabel('alpha')
ax_p.set_ylabel('beta')
ax_p.set_zlabel('gamma')
ax_p.legend()
plt.show()
