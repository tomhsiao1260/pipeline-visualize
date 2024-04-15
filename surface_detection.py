### Julian Schilliger - ThaumatoAnakalyptor - Vesuvius Challenge 2023

import os
import re
import sys
import tifffile
import torch
import numpy as np
from scipy.interpolate import interp1d

def umbilicus(points_array):
    """
    Interpolate between points in the provided 2D array based on z values.

    :param points_array: A 2D numpy array of shape (n, 3) with y, z, and x coordinates.
    :return: A 2D numpy array with interpolated points for each 1 step in the z direction.
    """

    # Separate the coordinates
    y, z, x = points_array.T

    # Create interpolation functions for x and y based on z
    fx = interp1d(z, x, kind='linear', fill_value="extrapolate")
    fy = interp1d(z, y, kind='linear', fill_value="extrapolate")

    # Define new z values for interpolation
    z_new = np.arange(z.min(), z.max(), 1)

    # Calculate interpolated x and y values
    x_new = fx(z_new)
    y_new = fy(z_new)

    # Return the combined y, z, and x values as a 2D array
    return np.column_stack((y_new, z_new, x_new))

def umbilicus_xz_at_y(points_array, y_new):
    """
    Interpolate between points in the provided 2D array based on y values.

    :param points_array: A 2D numpy array of shape (n, 3) with x, y, and z coordinates.
    :return: A 2D numpy array with interpolated points for each 0.1 step in the y direction.
    """

    # Separate the coordinates
    x, y, z = points_array.T

    # Create interpolation functions for x and z based on y
    fx = interp1d(y, x, kind='linear', fill_value="extrapolate")
    fz = interp1d(y, z, kind='linear', fill_value="extrapolate")

    # Calculate interpolated x and z values
    x_new = fx(y_new)
    z_new = fz(y_new)

    # Return the combined x, y, and z values as a 2D array
    res = np.array([x_new, y_new, z_new])
    return res

def get_reference_vector(grid_block_size, umbilicus_points, corner_coords):
    block_point = np.array(corner_coords) + np.array(grid_block_size) // 2
    umbilicus_point = umbilicus_xz_at_y(umbilicus_points, block_point[2])
    umbilicus_point = umbilicus_point[[0, 2, 1]] # ply to corner coords
    umbilicus_normal = block_point - umbilicus_point
    umbilicus_normal = umbilicus_normal[[2, 0, 1]] # corner coords to tif
    unit_umbilicus_normal = umbilicus_normal / np.linalg.norm(umbilicus_normal)
    return unit_umbilicus_normal

def get_corner_coords(start_point, cell_block_name):
    matches = re.findall(r'\d+', cell_block_name)
    cell_indices = tuple(int(match) for match in matches)
    corner_coords = tuple(p + 500 * i for i, p in zip(cell_indices, start_point))
    return corner_coords

if __name__ == '__main__':
    args = sys.argv

    # Check file path
    if len(args) < 2 or not args[1].endswith(".tif"):
        raise ValueError('TIFF path is not found')
        sys.exit(1)

    # Region you want to see (z, y, x)
    start_point = (0, 0, 0)
    box_size = (300, 300, 300)

    # Load tif data
    path = args[1]
    volume = tifffile.imread(path)
    # Crop the volume
    sz, sy, sx = start_point
    bz, by, bx = box_size
    volume = volume[sz:sz+bz, sy:sy+by, sx:sx+bx]
    # Convert to float32 tensor
    volume = np.uint8(volume//256)
    volume = torch.from_numpy(volume).float()

    # Load umbilicus data
    umbilicus_path = 'umbilicus.txt'
    umbilicus_raw_points = np.loadtxt(umbilicus_path, delimiter=',')
    umbilicus_points = umbilicus(umbilicus_raw_points)

    # Get corner coordinate
    cell_block_name = os.path.basename(path)
    start_point = (start_point[1], start_point[2], start_point[0])
    corner_coords = get_corner_coords(start_point, cell_block_name)
    # Calculate reference vector
    grid_block_size = volume.permute(1, 2, 0).shape
    global_reference_vector = get_reference_vector(grid_block_size, umbilicus_points, corner_coords)


