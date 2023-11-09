import torch
import PIL.Image
import numpy as np

from matplotlib import pyplot as plt

def pixel_wise_loss_function(_x, _target, _scale=1e3, _dims=[]):
    return _scale * torch.mean(torch.square(_x[:, :4] - _target), _dims)

def logprint(_path, _str):
    print (_str)
    with open(_path, 'a') as f:
        f.write(f'{_str}\n')
        
def create_seed(_size=16, _channels=16, _dist=5, _points=4, _last_channel=None, _angle=0.0):
    pass

# * loads an image and converts to a tensor
# *     default tensor shape: [BATCH_SIZE (1), CHANNELS (4), WIDTH (_size), HEIGHT (_size)]
def load_image_as_tensor(_path, _size, _resample=PIL.Image.Resampling.BICUBIC):
    img = PIL.Image.open(_path)
    img = img.resize((_size, _size), _resample)
    img = np.float32(img) / 255.0
    img[..., :3] *= img[..., 3:]
    return torch.from_numpy(img).permute(2, 0, 1)[None, ...]

# * given a tensor of default shape, visualize the first 4 channels as a RGBA image
def show_tensor_as_image(_tensor):
    img = to_rgb(_tensor).squeeze().permute(1, 2, 0)
    plt.axis('off')
    plt.imshow(img)
    plt.show()

# * takes the first 4 channels of a tensor of default shape and converts to a RGB image
def to_rgb(_x, _alpha='BLACK'):
    rgb, a = _x[:, :3], _x[:, 3:4]
    if _alpha == 'BLACK':
        return torch.clamp(rgb, 0.0, 1.0)
    elif _alpha == 'WHITE':
        return torch.clamp(1.-a + rgb, 0.0, 1.0)

# * creates a circle mask centered at a position of a given radius
def circle_mask(_size, _radius, _pos):
    Y, X = np.ogrid[:_size, :_size]
    dist_from_center = np.sqrt((X - _pos[0])**2 + (Y-_pos[1])**2)
    mask = dist_from_center >= _radius
    return mask