import os
import cv2
import torch
import tifffile
import numpy as np
from scipy.ndimage import binary_dilation

def tif_to_video(video_dir, tif_path, time=15, repeats=1):
    if not os.path.exists(videoDir): os.makedirs(videoDir)

    tif_name = os.path.basename(tif_path)
    video_name = tif_name.replace('.tif', '.mp4')
    video_path = os.path.join(video_dir, video_name)

    # Load tif data
    data = tifffile.imread(tif_path)
    data = np.repeat(data, repeats=repeats, axis=0)
    data = np.repeat(data, repeats=repeats, axis=1)
    data = np.repeat(data, repeats=repeats, axis=2)
    d, h, w = data.shape[:3]

    # Create video writer
    fps = d / time
    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    writer = cv2.VideoWriter(video_path, fourcc, fps, (w, h))

    # Save as video
    for layer in range(d):
        image = data[layer, :, :]
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        cv2.imshow(tif_name, image)
        cv2.waitKey(int(1000 / fps))
        cv2.destroyAllWindows()
        writer.write(image)
    writer.release()

def torch_to_tif(tensor, path):
    volume = tensor.numpy()
    volume = np.abs(volume)
    volume = volume.astype(np.uint8)
    tifffile.imwrite(path, volume)

def save_tif():
    # Blur
    tensor = torch.load('output/blurred_volume.pt')
    torch_to_tif(tensor[1:-1], 'output/blurred_volume.tif')

    # Sobel
    tensor = torch.load('output/sobel_vectors.pt')
    torch_to_tif(tensor[1:-1], 'output/sobel_vectors.tif')

    # Sobel Sampling
    tensor = torch.load('output/sobel_vectors_subsampled.pt')
    torch_to_tif(tensor[1:-1], 'output/sobel_vectors_subsampled.tif')

    # Apply vector convolution to the Sobel vectors
    tensor = torch.load('output/vector_conv.pt') * 255
    torch_to_tif(tensor[1:-1], 'output/vector_conv.tif')

    # Adjust to the global direction
    tensor = torch.load('output/adjusted_vectors.pt') * 255
    torch_to_tif(tensor[1:-1], 'output/adjusted_vectors.tif')

    # Interpolate the adjusted vectors to the original size
    tensor = torch.load('output/adjusted_vectors_interp.pt') * 255
    torch_to_tif(tensor[1:-1], 'output/adjusted_vectors_interp.tif')

    # First Derivative (Green: > 0, Red: < 0)
    fd = torch.load('output/first_derivative.pt') * 255
    tensor = torch.zeros(fd.shape + (3,))
    tensor[..., 0][fd < 0] = fd[fd < 0]
    tensor[..., 1][fd > 0] = fd[fd > 0]
    torch_to_tif(tensor[1:-1], 'output/first_derivative.tif')

    # Second Derivative (Green: > 0, Red: < 0)
    sd = torch.load('output/second_derivative.pt') * 255
    tensor = torch.zeros(sd.shape + (3,))
    tensor[..., 0][sd < 0] = sd[sd < 0]
    tensor[..., 1][sd > 0] = sd[sd > 0]
    torch_to_tif(tensor[1:-1], 'output/second_derivative.tif')

    # Point Cloud (Green: Recto, Red: Verso)
    origin = tifffile.imread('output/origin.tif')[1:-1]
    recto = torch.load('output/mask_recto.pt')[1:-1]
    verso = torch.load('output/mask_verso.pt')[1:-1]
    recto = binary_dilation(recto, iterations=2)
    verso = binary_dilation(verso, iterations=2)
    recto_verso = np.stack([origin] * 3, axis=-1)
    recto_verso[verso] = np.array([255, 0, 0])
    recto_verso[recto] = np.array([0, 255, 0])
    tifffile.imwrite('output/recto_verso.tif', recto_verso)

if __name__ == '__main__':
    # Use tif format to visualize torch tensors
    save_tif()

    # Save tif into video
    videoDir = 'output/video'

    tif_to_video(videoDir, 'output/origin.tif')
    tif_to_video(videoDir, 'output/blurred_volume.tif')
    tif_to_video(videoDir, 'output/sobel_vectors.tif')
    tif_to_video(videoDir, 'output/sobel_vectors_subsampled.tif', repeats=10)
    tif_to_video(videoDir, 'output/adjusted_vectors_interp.tif')
    tif_to_video(videoDir, 'output/first_derivative.tif')
    tif_to_video(videoDir, 'output/second_derivative.tif')
    tif_to_video(videoDir, 'output/recto_verso.tif')
