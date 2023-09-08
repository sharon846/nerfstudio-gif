import json
import numpy as np
import matplotlib.pyplot as plt
import time
import os
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import cv2

def discard_point(matrix_values, index):
    result_array = np.concatenate((matrix_values[:index], matrix_values[index+1:]))
    x = result_array[:,0,3]
    y = result_array[:,1,3]
    z = result_array[:,2,3]
    look_at = result_array[:,0:3,2]
    return x, y, z, look_at

def save_viewer(output_dir, origin, removed_cycle, removed_spline, new_generated):

	fig = make_subplots(rows=1, cols=1, specs=[[{'type': 'scatter3d'}]])
	scatter1 = go.Scatter3d(
		x=origin[:,0],
		y=origin[:,1],
		z=origin[:,2],
		mode='markers',
		marker=dict(size=6, color='green'),
		name='All points'
	)
	
	if len(removed_cycle) > 0:
		scatter2 = go.Scatter3d(
			x=removed_cycle[:,0],
			y=removed_cycle[:,1],
			z=removed_cycle[:,2],
			mode='markers',
			marker=dict(size=6, color='black'),
			name='Removed on loop detection'
		)
		fig.add_trace(scatter2)
	
	if len(removed_spline) > 0:
		scatter3 = go.Scatter3d(
			x=removed_spline[:,0],
			y=removed_spline[:,1],
			z=removed_spline[:,2],
			mode='markers',
			marker=dict(size=6, color='red'),
			name='Removed on outliers detection'
		)
		fig.add_trace(scatter3)
	
	scatter4 = go.Scatter3d(
		x=new_generated[:,0],
		y=new_generated[:,1],
		z=new_generated[:,2],
		mode='markers',
		marker=dict(size=6, color='blue'),
		name='Added points'
	)
	
	fig.add_trace(scatter1)
	fig.add_trace(scatter4)
	
	fig.update_layout(
		scene=dict(
			xaxis_title='X-axis',
			yaxis_title='Y-axis',
			zaxis_title='Z-axis'
		),
		title='camera 3d viewer',
	)

	# Save the figure as an HTML file
	fig.write_html(f"{output_dir}/points_viewer.html")

def scatter_data(ax, origin, removed_cycle, removed_spline, new_generated):
	ax.cla()
	ax.scatter(origin[:,0], origin[:,1], origin[:,2], color='green', label='All points')
	if len(removed_cycle) > 0:
		ax.scatter(removed_cycle[:,0], removed_cycle[:,1], removed_cycle[:,2], color='black', label='Removed on loop detection')
	if len(removed_spline) > 0:
		ax.scatter(removed_spline[:,0], removed_spline[:,1], removed_spline[:,2], color='red', label='Removed on outliers detection')
	ax.scatter(new_generated[:,0], new_generated[:,1], new_generated[:,2],color='blue', label='Added points')

def save_camera(output_dir, camera_path, origin, removed_cycle, removed_spline, new_generated):

	fig = plt.figure(figsize=(10, 8))
	ax = fig.add_subplot(111, projection='3d')  # 3D plot

	ax.set_xlabel('X')
	ax.set_ylabel('Y')
	ax.set_zlabel('Z')
	
	with open(camera_path, 'r') as f:
		data = json.load(f)

	matrix_values = []

	for keyframe in data['keyframes']:
		if 'matrix' in keyframe:
			matrix_str = keyframe['matrix']
			try:
				matrix_array = np.array(json.loads(matrix_str)).reshape((4,4)).T
				matrix_values.append(matrix_array)
			except (ValueError, json.JSONDecodeError):
				print(f"Unable to parse 'matrix' value: {matrix_str}")
	
	matrix_values = np.array(matrix_values)
	poses = np.concatenate((origin, new_generated), axis=0)
	x = poses[:,0]
	y = poses[:,1]
	z = poses[:,2]
	look_at = matrix_values[:,0:3,2]

	for i in range(len(matrix_values)):
		print(f'Step {i:03d}', end='\r')
		scatter_data(ax, origin, removed_cycle, removed_spline, new_generated)
		ax.scatter(x[i],y[i],z[i],color='purple')
		ax.quiver(*(x[i],y[i],z[i]), -2*look_at[i][0], -2*look_at[i][1], -2*look_at[i][2], color='purple')
		fig.savefig(f"{output_dir}/plot{i:05d}.png")
