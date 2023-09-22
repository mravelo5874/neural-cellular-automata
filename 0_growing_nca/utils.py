# imports
import numpy as np
import torch
import PIL.Image, PIL.ImageDraw

# utility function class
class Utils:
    # creates a circle mask given a size, radius and position
    def create_circle_mask(size, radius, pos):
        Y, X = np.ogrid[:size, :size]
        dist_from_center = np.sqrt((X - pos[0])**2 + (Y-pos[1])**2)
        mask = dist_from_center >= radius
        return mask
    
    def create_erase_mask(size, radius, pos):
        pos = pos * size
        mask = Utils.create_circle_mask(size, radius, pos)
        return mask

    # Loads an image from a specified path and converts to torch.Tensor
    def load_image(path, size):
        img = PIL.Image.open(path)
        img = img.resize((size, size), PIL.Image.Resampling.BILINEAR)
        img = np.float32(img) / 255.0
        img[..., :3] *= img[..., 3:]
        return torch.from_numpy(img).permute(2, 0, 1)[None, ...]

    # converts an RGBA image (tensor) to an RGB image (tensor)
    def to_rgb(img_rgba):
        rgb, a = img_rgba[:, :3, ...], torch.clamp(img_rgba[:, 3:, ...], 0, 1)
        return torch.clamp(rgb, 0, 1)
    
    # converts a tensor to a RGBA image (tensor)
    def to_rgba(x):
        x = x[:, :4, ...]
        return x

    # Create a starting tensor for training
    # Only the active pixels are goin to be in the middle
    def make_seed(size, n_channels):
        x = torch.zeros((1, n_channels, size, size), dtype=torch.float32)
        x[:, 3:, size // 2, size // 2] = 1
        return x