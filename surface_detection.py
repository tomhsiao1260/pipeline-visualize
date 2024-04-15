### Julian Schilliger - ThaumatoAnakalyptor - Vesuvius Challenge 2023

import torch
import torch.nn as nn
import torch.nn.functional as F

import os
import re
import sys
import tifffile
import numpy as np
import open3d as o3d
from scipy.interpolate import interp1d

def save_ply(recto_tensor_tuple, verso_tensor_tuple, corner_coords, grid_block_size, padding):
    points_r_tensor, normals_r_tensor = recto_tensor_tuple
    points_v_tensor, normals_v_tensor = verso_tensor_tuple

    # Extract actual volume size from the oversized input block
    grid_block_position_min = [0 + padding for size in grid_block_size]
    grid_block_position_max = [size + padding for size in grid_block_size]
    points_r_tensor, normals_r_tensor = extract_size_tensor(points_r_tensor, normals_r_tensor, grid_block_position_min, grid_block_position_max) # 0, 0, 0 is the minimum corner of the grid block
    points_v_tensor, normals_v_tensor = extract_size_tensor(points_v_tensor, normals_v_tensor, grid_block_position_min, grid_block_position_max) # 0, 0, 0 is the minimum corner of the grid block

    ### Adjust the 3D coordinates of the points based on their position in the larger volume

    # permute the axes to match the original volume
    points_r_tensor = points_r_tensor[:, [1, 0, 2]]
    normals_r_tensor = normals_r_tensor[:, [1, 0, 2]]
    points_v_tensor = points_v_tensor[:, [1, 0, 2]]
    normals_v_tensor = normals_v_tensor[:, [1, 0, 2]]

    y_d, x_d, z_d = corner_coords
    points_r_tensor += torch.tensor([y_d, z_d, x_d], dtype=points_r_tensor.dtype, device=points_r_tensor.device)
    points_v_tensor += torch.tensor([y_d, z_d, x_d], dtype=points_v_tensor.dtype, device=points_v_tensor.device)

    # Save the surface points and normals as a PLY file
    points_r = points_r_tensor.cpu().numpy()
    normals_r = normals_r_tensor.cpu().numpy()
    points_v = points_v_tensor.cpu().numpy()
    normals_v = normals_v_tensor.cpu().numpy()

    save_template_v = "output/point_cloud_recto/" + "cell_yxz_{:03}_{:03}_{:03}.ply"
    save_template_r = "output/point_cloud_verso/" + "cell_yxz_{:03}_{:03}_{:03}.ply"

    file_x, file_y, file_z = corner_coords[0]//grid_block_size[0], corner_coords[1]//grid_block_size[1], corner_coords[2]//grid_block_size[2]
    surface_ply_filename_v = save_template_v.format(file_x, file_y, file_z)
    surface_ply_filename_r = save_template_r.format(file_x, file_y, file_z)

    save_surface_ply(points_r, normals_r, surface_ply_filename_r)
    save_surface_ply(points_v, normals_v, surface_ply_filename_v)
    print('Save Recto Point Cloud ', points_r.shape[0])
    print('Save Verso Point Cloud ', points_v.shape[0])

def extract_size_tensor(points, normals, grid_block_position_min, grid_block_position_max):
    """
    Extract points and corresponding normals that lie within the given size range.

    Parameters:
        points (torch.Tensor): The point coordinates, shape (n, 3).
        normals (torch.Tensor): The point normals, shape (n, 3).
        grid_block_position_min (list): The minimum block size, shape (3,).
        grid_block_position_max (list): The maximum block size, shape (3,).

    Returns:
        filtered_points (torch.Tensor): The filtered points, shape (m, 3).
        filtered_normals (torch.Tensor): The corresponding filtered normals, shape (m, 3).
    """

    # Convert min and max to tensors for comparison
    min_tensor = torch.tensor(grid_block_position_min, dtype=points.dtype, device=points.device)
    max_tensor = torch.tensor(grid_block_position_max, dtype=points.dtype, device=points.device)

    # Create a mask to filter points within the specified range
    mask_min = torch.all(points >= min_tensor, dim=-1)
    mask_max = torch.all(points <= max_tensor, dim=-1)

    # Combine the masks to get the final mask
    mask = torch.logical_and(mask_min, mask_max)

    # Apply the mask to filter points and corresponding normals
    filtered_points = points[mask]
    filtered_normals = normals[mask]

    # Reposition the points to be relative to the grid block
    filtered_points -= min_tensor

    return filtered_points, filtered_normals

def save_surface_ply(surface_points, normals, filename, color=None):
    try:
        if (len(surface_points)  < 1):
            return
        # Create an Open3D point cloud object and populate it
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(surface_points.astype(np.float32))
        pcd.normals = o3d.utility.Vector3dVector(normals.astype(np.float16))
        if color is not None:
            pcd.colors = o3d.utility.Vector3dVector(color.astype(np.float16))

        # Create folder if it doesn't exist
        os.makedirs(os.path.dirname(filename), exist_ok=True)

        # Save to a temporary file first to ensure data integrity
        temp_filename = filename.replace(".ply", "_temp.ply")
        o3d.io.write_point_cloud(temp_filename, pcd)

        # Rename the temp file to the original filename
        os.rename(temp_filename, filename)
    except Exception as e:
        print(f"Error saving surface PLY: {e}")

## sobel_filter_3d from https://github.com/lukeboi/scroll-viewer/blob/dev/server/app.py
### adjusted for my use case and improve efficiency
def sobel_filter_3d(input, chunks=4, overlap=3, device=None):
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input = input.unsqueeze(0).unsqueeze(0).to(device, dtype=torch.float16)

    # Define 3x3x3 kernels for Sobel operator in 3D
    sobel_x = torch.tensor([
        [[[ 1, 0, -1], [ 2, 0, -2], [ 1, 0, -1]],
         [[ 2, 0, -2], [ 4, 0, -4], [ 2, 0, -2]],
         [[ 1, 0, -1], [ 2, 0, -2], [ 1, 0, -1]]],
    ], dtype=torch.float16).to(device)

    sobel_y = sobel_x.transpose(2, 3)
    sobel_z = sobel_x.transpose(1, 3)

    # Add an extra dimension for the input channels
    sobel_x = sobel_x[None, ...]
    sobel_y = sobel_y[None, ...]
    sobel_z = sobel_z[None, ...]

    assert len(input.shape) == 5, "Expected 5D input (batch_size, channels, depth, height, width)"

    depth = input.shape[2]
    chunk_size = depth // chunks
    chunk_overlap = overlap // 2

    # Initialize tensors for results and vectors if needed
    vectors = torch.zeros(list(input.shape) + [3], device=device, dtype=torch.float16)

    for i in range(chunks):
        # Determine the start and end index of the chunk
        start = max(0, i * chunk_size - chunk_overlap)
        end = min(depth, (i + 1) * chunk_size + chunk_overlap)

        if i == chunks - 1:  # Adjust the end index for the last chunk
            end = depth

        chunk = input[:, :, start:end, :, :]

        # Move chunk to GPU
        chunk = chunk.to(device, non_blocking=True)  # Use non_blocking transfers

        G_x = nn.functional.conv3d(chunk, sobel_x, padding=1)
        G_y = nn.functional.conv3d(chunk, sobel_y, padding=1)
        G_z = nn.functional.conv3d(chunk, sobel_z, padding=1)

        # Overlap removal can be optimized
        actual_start = 0 if i == 0 else chunk_overlap
        actual_end = -chunk_overlap if i != chunks - 1 else None
        # Stack gradients in-place if needed
        vectors[:, :, start + actual_start:end + (actual_end if actual_end is not None else 0), :, :] = torch.stack((G_x, G_y, G_z), dim=5)[:, :, actual_start:actual_end, :, :]

        # Free memory of intermediate variables
        del G_x, G_y, G_z, chunk
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return vectors.squeeze(0).squeeze(0).to(device)

# Function to create a 3D Uniform kernel
def get_uniform_kernel(size=3, channels=1):
    # Create a 3D kernel filled with ones and normalize it
    kernel = torch.ones((size, size, size))
    kernel = kernel / torch.sum(kernel)
    return kernel

# Function to create a 3D convolution layer with a Uniform kernel
def uniform_blur3d(channels=1, size=3, device=None):
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    kernel = get_uniform_kernel(size, channels)
    # Repeat the kernel for all input channels
    kernel = kernel.repeat(channels, 1, 1, 1, 1)
    # Create a convolution layer
    blur_layer = nn.Conv3d(in_channels=channels, out_channels=channels, 
                           kernel_size=size, groups=channels, bias=False, padding=size//2)
    # Set the kernel weights
    blur_layer.weight.data = nn.Parameter(kernel)
    # Make the layer non-trainable
    blur_layer.weight.requires_grad = False
    blur_layer.to(device)
    return blur_layer

# Function to normalize vectors to unit length
def normalize(vectors):
    return vectors / vectors.norm(dim=-1, keepdim=True)

# Function to calculate angular distance between vectors
def angular_distance(v1, v2):
    v1 = v1.unsqueeze(0) if v1.dim() == 1 else v1
    v2 = v2.unsqueeze(0) if v2.dim() == 1 else v2
    return torch.acos(torch.clamp((v1 * v2).sum(-1), -1.0, 1.0))

# Mean indiscriminative Loss function for a batch of candidate vectors and a batch of input vectors
def loss(candidates, vectors):
    vector_normed = normalize(vectors)
    pos_distance = angular_distance(candidates[:, None, :], vector_normed[None, :, :])
    neg_distance = angular_distance(-candidates[:, None, :], vector_normed[None, :, :])
    losses = torch.min(torch.stack((pos_distance, neg_distance), dim=-1), dim=-1)[0]
    
    # calculate the norm of the vectors and use it to scale the losses
    vector_norms = torch.norm(vectors, dim=-1)
    scaled_losses = losses * vector_norms
    
    return scaled_losses

# Function to find the vector that is the propper mean vector for the input vectors vs when -v = v
def find_mean_indiscriminative_vector(vectors, n, device):
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Normalize the input vectors
    vectors = normalize(vectors)
    
    # Generate n random unit vectors
    random_vectors = torch.randn(n, 3, device=device)
    random_vectors = normalize(random_vectors)

    # Compute the total loss for each candidate
    total_loss = loss(random_vectors, vectors).sum(dim=-1)

    # Find the best candidate
    best_vector = random_vectors[torch.argmin(total_loss)]

    return best_vector

# Function that  projects vector a onto vector b
def vector_projection(a, b):
    return (a * b).sum(-1, keepdim=True) * b / b.norm(dim=-1, keepdim=True)**2

# Function that adjusts the norm of vector a to the norm of vector b based on their direction
def adjusted_norm(a, b):
    # Calculate the projection
    projection = vector_projection(a, b)
    
    # Compute the dot product of the projected vector and the original vector b
    dot_product = (projection * b).sum(-1, keepdim=True)
    
    # Compute the norm of the projection
    projection_norm = projection.norm(dim=-1)
    
    # Adjust the norm based on the sign of the dot product
    adjusted_norm = torch.sign(dot_product.squeeze()) * projection_norm
    
    return adjusted_norm

def scale_to_0_1(tensor):
    # Compute the 99th percentile
    #quantile_val = torch.quantile(tensor, 0.95)
    
    # Clip the tensor values at the 99th percentile
    clipped_tensor = torch.clamp(tensor, min=-1000, max=1000)
    
    # Scale the tensor to the range [0,1]
    tensor_min = torch.min(clipped_tensor)
    tensor_max = torch.max(clipped_tensor)
    tensor_scale = torch.max(torch.abs(tensor_min), torch.abs(tensor_max))
    scaled_tensor = clipped_tensor / tensor_scale
    return scaled_tensor

# Function that convolutes a 3D Volume of vectors to find their mean indiscriminative vector
def vector_convolution(input_tensor, window_size=20, stride=20, device=None):
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_tensor = input_tensor.to(device)
    # get the size of your 4D input tensor
    input_size = input_tensor.size()

    # initialize an output tensor
    output_tensor = torch.zeros((input_size[0] - window_size + 1) // stride, 
                                (input_size[1] - window_size + 1) // stride, 
                                (input_size[2] - window_size + 1) // stride, 
                                3, device=device)

    # slide the window across the 3D volume
    for i in range(0, input_size[0] - window_size + 1, stride):
        for j in range(0, input_size[1] - window_size + 1, stride):
            for k in range(0, input_size[2] - window_size + 1, stride):
                # extract the vectors within the window
                window_vectors = input_tensor[i:i+window_size, j:j+window_size, k:k+window_size]
                window_vectors = window_vectors.reshape(-1, 3)  # flatten the 3D window into a 2D tensor
                
                # calculate the closest vector
                best_vector = find_mean_indiscriminative_vector(window_vectors, 100, device)  # adjust the second parameter as needed
                # check if the indices are within the output_tensor's dimension
                if i//stride < output_tensor.shape[0] and j//stride < output_tensor.shape[1] and k//stride < output_tensor.shape[2]:
                    # store the result in the output tensor
                    output_tensor[i//stride, j//stride, k//stride] = best_vector

    return output_tensor

# Function that interpolates the output tensor to the original size of the input tensor
def interpolate_to_original(input_tensor, output_tensor):
    # Adjust the shape of the output tensor to match the input tensor
    # by applying 3D interpolation. We're assuming that the last dimension
    # of both tensors is the channel dimension (which should not be interpolated over).
    output_tensor = output_tensor.permute(3, 0, 1, 2)
    input_tensor = input_tensor.permute(3, 0, 1, 2)

    # Use 3D interpolation to resize output_tensor to match the shape of input_tensor.
    interpolated_tensor = F.interpolate(output_tensor.unsqueeze(0), size=input_tensor.shape[1:], mode='trilinear', align_corners=False)

    # Return the tensor to its original shape.
    interpolated_tensor = interpolated_tensor.squeeze(0).permute(1, 2, 3, 0)

    return interpolated_tensor

# Function that adjusts the vectors in the input tensor to point in the same general direction as the global reference vector.
def adjust_vectors_to_global_direction(input_tensor, global_reference_vector):
    # Compute dot product of each vector in the input tensor with the global reference vector.
    # The resulting tensor will have the same shape as the input tensor, but with the last dimension squeezed out.
    dot_products = (input_tensor * global_reference_vector).sum(dim=-1, keepdim=True)
    # Create a mask of the same shape as the dot products tensor, 
    # with True wherever the dot product is negative.
    mask = dot_products < 0
    
    # Expand the mask to have the same shape as the input tensor
    mask = mask.expand(input_tensor.shape)

    # Negate all vectors in the input tensor where the mask is True.
    adjusted_tensor = input_tensor.clone()
    adjusted_tensor[mask] = -input_tensor[mask]

    return adjusted_tensor

# Function to detect surface points in a 3D volume
def surface_detection(volume, global_reference_vector, blur_size=3, sobel_chunks=4, sobel_overlap=3, window_size=20, stride=20, threshold_der=0.1, threshold_der2=0.001, convert_to_numpy=True):
    # device
    print('Processing ...')
    device = volume.device

    # using half percision to save memory
    volume = volume
    # Blur the volume
    blur = uniform_blur3d(channels=1, size=blur_size, device=device)
    blurred_volume = blur(volume.unsqueeze(0).unsqueeze(0)).squeeze(0).squeeze(0)
    torch.save(blurred_volume, 'output/blurred_volume.pt')
    print('Blur ', '({}, {}, {})'.format(*blurred_volume.shape))

    # Apply Sobel filter to the blurred volume
    sobel_vectors = sobel_filter_3d(volume, chunks=sobel_chunks, overlap=sobel_overlap, device=device)
    torch.save(sobel_vectors, 'output/sobel_vectors.pt')
    print('Sobel ', '({}, {}, {})'.format(*sobel_vectors.shape))

    # Subsample the sobel_vectors
    sobel_stride = 10
    sobel_vectors_subsampled = sobel_vectors[::sobel_stride, ::sobel_stride, ::sobel_stride, :]
    torch.save(sobel_vectors_subsampled, 'output/sobel_vectors_subsampled.pt')
    print('Sobel Sampling ', '({}, {}, {})'.format(*sobel_vectors_subsampled.shape))

    # Apply vector convolution to the Sobel vectors
    vector_conv = vector_convolution(sobel_vectors_subsampled, window_size=window_size, stride=stride, device=device)
    torch.save(vector_conv, 'output/vector_conv.pt')
    print('Sobel Vector Convolution ', '({}, {}, {})'.format(*vector_conv.shape))

    # Adjust vectors to the global direction
    adjusted_vectors = adjust_vectors_to_global_direction(vector_conv, global_reference_vector)
    torch.save(adjusted_vectors, 'output/adjusted_vectors.pt')
    print('Adjust to Global Direction ', '({}, {}, {})'.format(*adjusted_vectors.shape))

    # Interpolate the adjusted vectors to the original size
    adjusted_vectors_interp = interpolate_to_original(sobel_vectors, adjusted_vectors)
    torch.save(adjusted_vectors_interp, 'output/adjusted_vectors_interp.pt')
    print('Adjust to Original Size ', '({}, {}, {})'.format(*adjusted_vectors_interp.shape))

    # Project the Sobel result onto the adjusted vectors and calculate the norm
    first_derivative = adjusted_norm(sobel_vectors, adjusted_vectors_interp)
    fshape = first_derivative.shape
    
    first_derivative = scale_to_0_1(first_derivative)
    torch.save(first_derivative, 'output/first_derivative.pt')
    print('First Derivative ', '({}, {}, {})'.format(*first_derivative.shape))

    # Apply Sobel filter to the first derivative, project it onto the adjusted vectors, and calculate the norm
    sobel_vectors_derivative = sobel_filter_3d(first_derivative, chunks=sobel_chunks, overlap=sobel_overlap, device=device)
    second_derivative = adjusted_norm(sobel_vectors_derivative, adjusted_vectors_interp)
    second_derivative = scale_to_0_1(second_derivative)
    torch.save(second_derivative, 'output/second_derivative.pt')
    print('Second Derivative ', '({}, {}, {})'.format(*second_derivative.shape))

    # Generate recto side of sheet

    # Create a mask for the conditions on the first and second derivatives
    mask_recto = (second_derivative.abs() < threshold_der2) & (first_derivative > threshold_der)
    torch.save(mask_recto, 'output/mask_recto.pt')
    print('Mask of Recto ', '({}, {}, {})'.format(*mask_recto.shape))
    # Check where the second derivative is zero and the first derivative is above a threshold
    points_to_mark = torch.where(mask_recto)

    # Subsample the points to mark
    #subsample_nr = 2000000
    coords = torch.stack(points_to_mark, dim=1)
    #coords = subsample_uniform(coords, subsample_nr)
    
    # Cluster the surface points
    coords_normals = adjusted_vectors_interp[coords[:, 0], coords[:, 1], coords[:, 2]]
    coords_normals = coords_normals / torch.norm(coords_normals, dim=1, keepdim=True)

    # Generate verso side of sheet
    # Create a mask for the conditions on the first and second derivatives
    mask_verso = (second_derivative.abs() < threshold_der2) & (first_derivative < -threshold_der)
    torch.save(mask_verso, 'output/mask_verso.pt')
    print('Mask of Verso ', '({}, {}, {})'.format(*mask_verso.shape))
    # Check where the second derivative is zero and the first derivative is above a threshold
    points_to_mark_verso = torch.where(mask_verso)

    coords_verso = torch.stack(points_to_mark_verso, dim=1)
    
    # Cluster the surface points
    coords_normals_verso = adjusted_vectors_interp[coords_verso[:, 0], coords_verso[:, 1], coords_verso[:, 2]]
    coords_normals_verso = coords_normals_verso / torch.norm(coords_normals_verso, dim=1, keepdim=True)

    if convert_to_numpy:
        coords = coords.cpu().numpy()
        coords_normals = coords_normals.cpu().numpy()
        
        coords_verso = coords_verso.cpu().numpy()
        coords_normals_verso = coords_normals_verso.cpu().numpy()

    return (coords, coords_normals), (coords_verso, coords_normals_verso)

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
    if not os.path.exists('output'): os.makedirs('output')
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
    # Save the volume
    volume = np.uint8(volume//256)
    tifffile.imwrite('output/origin.tif', volume)
    # Convert to float32 tensor
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

    # Main calculation here
    recto_tensor_tuple, verso_tensor_tuple = surface_detection(volume, global_reference_vector, blur_size=11, window_size=9, stride=1, threshold_der=0.5, threshold_der2=0.002, convert_to_numpy=False)

    # Save point cloud
    save_ply(recto_tensor_tuple, verso_tensor_tuple, corner_coords, grid_block_size, padding=0)


