import torch
import numpy as np
import matplotlib.gridspec as gridspec
from colorsys import hsv_to_rgb
from matplotlib import pyplot as plt
from scripts.vox.Vox import Vox

def logprint(_path, _str):
    print (_str)
    with open(_path, 'a', encoding='utf-8') as f:
        f.write(f'{_str}\n')
        
def voxel_wise_loss_function(_x, _target, _scale=1e3, _dims=[]):
    return _scale * np.mean(np.square(_x[:, :4] - _target), _dims)

# * shows a batch before and after a forward pass given two (2) tensors
def show_batch(_batch_size, _before, _after, _dpi=256):
    fig = plt.figure(figsize=(_batch_size, 2), dpi=_dpi)
    axarr = fig.subplots(nrows=2, ncols=_batch_size)
    gspec = gridspec.GridSpec(2, _batch_size)
    gspec.update(wspace=0, hspace=0) # set the spacing between axes.
    plt.clf()
    for i in range(_batch_size):
        vox = Vox().load_from_tensor(_before[i, ...])
        img = vox.render(_print=False)
        axarr[0, i] = plt.subplot(gspec[i])
        axarr[0, i].set_xticks([])
        axarr[0, i].set_yticks([])
        axarr[0, i].imshow(img, aspect='equal')
        axarr[0, i].set_title(str(i), fontsize=8)   
    for i in range(_batch_size):
        vox = Vox().load_from_tensor(_after[i, ...])
        img = vox.render(_print=False)
        axarr[1, i] = plt.subplot(gspec[i+_batch_size])
        axarr[1, i].set_xticks([])
        axarr[1, i].set_yticks([])
        axarr[1, i].imshow(img, aspect='equal')
    plt.show()

def create_seed(_size=16, _channels=16, _dist=5, _points=4):
    x = torch.zeros([_channels, _size, _size, _size])
    half = _size//2
    # * black
    if _points == 1:
        x[3:_channels, half, half, half] = 1.0
    else:
        # * red
        if _points > 0:
            x[3:_channels, half, half, half] = 1.0
            x[0, half, half, half] = 1.0
        # * green
        if _points > 1:
            x[3:_channels, half, half+_dist, half] = 1.0
            x[1, half, half+_dist, half] = 1.0
        # * blue
        if _points > 2:
            x[3:_channels, half+_dist, half, half] = 1.0
            x[2, half+_dist, half, half] = 1.0
        # * yellow
        if _points > 3:
            x[3:_channels, half, half, half+_dist] = 1.0
            x[0:2, half, half, half+_dist] = 1.0
        # * magenta
        if _points > 4:
            x[3:_channels, half, half-_dist, half] = 1.0
            x[0, half, half-_dist, half] = 1.0
            x[2, half, half-_dist, half] = 1.0
        # * cyan
        if _points > 5:
            x[3:_channels, half-_dist, half, half] = 1.0
            x[1:3, half-_dist, half, half] = 1.0
    return x

def color_to_channels(_color='BLACK'):
    color = _color.lower()
    if color == 'black':
        return []
    elif color == 'red':
        return [0]
    elif color == 'green':
        return [1]
    elif color == 'blue':
        return [2]
    elif color == 'yellow':
        return [0, 1]
    elif color == 'pink':
        return [0, 2]
    elif color == 'cyan':
        return [1, 2]
    elif color == 'white':
        return [0, 1, 2]

def custom_seed(_size=16, _channels=16, _dist=5, _hidden_info=False, 
                _center=None, 
                _plus_x=None, _minus_x=None, 
                _plus_y=None, _minus_y=None, 
                _plus_z=None, _minus_z=None):
    x = np.zeros([_channels, _size, _size, _size])
    half = _size//2
    
    if _center != None:
        chns = color_to_channels(_center)
        x[3:_channels, half, half, half] = 1.0
        for i in range(len(chns)):
            x[chns[i], half, half, half] = 1.0
        if _hidden_info:
            for i in range(len(chns)):
                x[chns[i]+4, half, half, half] = 0.0
            
    if _plus_x != None:
        chns = color_to_channels(_plus_x)
        x[3:_channels, half+_dist, half, half] = 1.0
        for i in range(len(chns)):
            x[chns[i], half+_dist, half, half] = 1.0
        if _hidden_info:
            for i in range(len(chns)):
                x[chns[i]+4, half+_dist, half, half] = 0.0
            
    if _minus_x != None:
        chns = color_to_channels(_minus_x)
        x[3:_channels, half-_dist, half, half] = 1.0
        for i in range(len(chns)):
            x[chns[i], half-_dist, half, half] = 1.0
        if _hidden_info:
            for i in range(len(chns)):
                x[chns[i]+4, half-_dist, half, half] = 0.0
    
    if _plus_y != None:
        chns = color_to_channels(_plus_y)
        x[3:_channels, half, half+_dist, half] = 1.0
        for i in range(len(chns)):
            x[chns[i], half, half+_dist, half] = 1.0
        if _hidden_info:
            for i in range(len(chns)):
                x[chns[i]+4, half, half+_dist, half] = 0.0
            
    if _minus_y != None:
        chns = color_to_channels(_minus_y)
        x[3:_channels, half, half-_dist, half] = 1.0
        for i in range(len(chns)):
            x[chns[i], half, half-_dist, half] = 1.0
        if _hidden_info:
            for i in range(len(chns)):
                x[chns[i]+4, half, half-_dist, half] = 0.0
            
    if _plus_z != None:
        chns = color_to_channels(_plus_z)
        x[3:_channels, half, half, half+_dist] = 1.0
        for i in range(len(chns)):
            x[chns[i], half, half, half+_dist] = 1.0
        if _hidden_info:
            for i in range(len(chns)):
                x[chns[i]+4, half, half, half+_dist] = 0.0
            
    if _minus_z != None:
        chns = color_to_channels(_minus_z)
        x[3:_channels, half, half, half-_dist] = 1.0
        for i in range(len(chns)):
            x[chns[i], half, half, half-_dist] = 1.0
        if _hidden_info:
            for i in range(len(chns)):
                x[chns[i]+4, half, half, half-_dist] = 0.0
    return x

def rgb_linspace(n):
    '''Generates n visually distinct rgb combinations'''
    return np.array([hsv_to_rgb(i / n, 1.0, 1.0) for i in range(n)])

def seed_3d(_size=128, _channels=16, _points=3, _radius=4, _xyz=None, _rgb_dist=rgb_linspace):
    '''Generates a uniform p-point structured seed of radius r in 3D'''
    x = np.zeros(_channels, _size, _size, _size)
    # Initialize p points equidistant around a sphere of radius r
    indices = np.arange(0, _points, dtype=float) + 0.5
    phi = (np.arccos(1 - 2*indices/_points))
    theta = np.pi * (1 + 5**0.5) * indices
    if _xyz is None:
        dx, dy, dz = (np.c_[_radius*np.cos(theta)*np.sin(phi), _radius*np.sin(theta)*np.sin(phi), _radius*np.cos(phi)]+(_size//2)).astype(np.int32).T
        _xyz = np.array([dx, dy, dz])
    # Assign distinct rgb values to each point
    x[0:3, _xyz[0], _xyz[1], _xyz[2]] =  _rgb_dist(_xyz.shape[1]).T
    x[3:_channels, _xyz[0], _xyz[1], _xyz[2]] = 1.0
    return x

def half_volume_mask(_size, _type):
    mask_types = ['x+', 'x-', 'y+', 'y-', 'z+', 'z-', 'rand']
    if _type == 'rand':
        _type = mask_types[np.random.randint(0, 6)]
    mat = np.zeros([_size, _size, _size])
    half = _size//2
    if _type == 'x+':
        mat[:half, :, :] = 1.0
    elif _type == 'x-':
        mat[-half:, :, :] = 1.0
    if _type == 'y+':
        mat[:, :half, :] = 1.0
    elif _type == 'y-':
        mat[:, -half:, :] = 1.0
    if _type == 'z+':
        mat[:, :, :half] = 1.0
    elif _type == 'z-':
        mat[:, :, -half:] = 1.0
    return mat > 0.0

def euler_to_quaternion(_ax, _ay, _az):
    # * get sin and cos values
    cx, sx = torch.cos(_ax/2), torch.sin(_ax/2)
    cy, sy = torch.cos(_ay/2), torch.sin(_ay/2)
    cz, sz = torch.cos(_az/2), torch.sin(_az/2)

    # * compute w, x, y, z
    w = cx*cy*cz+sx*sy*sz
    x = sx*cy*cz-cx*sy*sz
    y = cx*sy*cz+sx*cy*sz
    z = cx*cy*sz-sx*sy*cz
    
    # * return quat values
    return torch.cat([w, x, y, z], 1)