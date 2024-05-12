import torch
import torch.nn.functional as func
import numpy as np
import matplotlib.gridspec as gridspec
from colorsys import hsv_to_rgb
from matplotlib import pyplot as plt
# * custom imports
from vox.Vox import Vox

def logprint(_path, _str):
    print (_str)
    with open(_path, 'a', encoding='utf-8') as f:
        f.write(f'{_str}\n')
        
def voxel_wise_loss_function(_x, _target, _scale=1e3, _dims=[]):
    return _scale * torch.mean(torch.square(_x[:, :4] - _target), _dims)

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
    x = torch.zeros([_channels, _size, _size, _size])
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
    return torch.tensor([hsv_to_rgb(i / n, 1.0, 1.0) for i in range(n)], dtype=torch.float32)

def seed_3d(_size=128, _channels=16, _points=3, _radius=4, _xyz=None, _rgb_dist=rgb_linspace):
    '''Generates a uniform p-point structured seed of radius r in 3D'''
    x = torch.zeros(_channels, _size, _size, _size)
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

# * yoinked (and modified) from https://stackoverflow.com/questions/54616049/converting-a-rotation-matrix-to-euler-angles-and-back-special-case
def eul2rot(ax, ay, az):

    r11 = torch.cos(ay)*torch.cos(az)
    r12 = torch.sin(ax)*torch.sin(ay)*torch.cos(az) - torch.sin(az)*torch.cos(ax)
    r13 = torch.sin(ay)*torch.cos(ax)*torch.cos(az) + torch.sin(ax)*torch.sin(az)
    
    r21 = torch.sin(az)*torch.cos(ay)
    r22 = torch.sin(ax)*torch.sin(ay)*torch.sin(az) + torch.cos(ax)*torch.cos(az)
    r23 = torch.sin(ay)*torch.sin(az)*torch.cos(ax) - torch.sin(ax)*torch.cos(az)
    
    r31 = -torch.sin(ax)
    r32 = torch.sin(ax)*torch.cos(ay)
    r33 = torch.cos(ax)*torch.cos(ay)
    
    b, c = ax.shape
    R = torch.tensor(np.zeros([b, c, 3, 3]), dtype=torch.float)
    
    R[:, :, 0, 0] = r11
    R[:, :, 0, 1] = r12
    R[:, :, 0, 2] = r13
    
    R[:, :, 1, 0] = r21
    R[:, :, 1, 1] = r22
    R[:, :, 1, 2] = r23
    
    R[:, :, 2, 0] = r31
    R[:, :, 2, 1] = r32
    R[:, :, 2, 2] = r33

    return R

def euler2rot_np(ax, ay, az):
    r11 = np.cos(ay)*np.cos(az)
    r12 = np.sin(ax)*np.sin(ay)*np.cos(az) - np.sin(az)*np.cos(ax)
    r13 = np.sin(ay)*np.cos(ax)*np.cos(az) + np.sin(ax)*np.sin(az)
    
    r21 = np.sin(az)*np.cos(ay)
    r22 = np.sin(ax)*np.sin(ay)*np.sin(az) + np.cos(ax)*np.cos(az)
    r23 = np.sin(ay)*np.sin(az)*np.cos(ax) - np.sin(ax)*np.cos(az)
    
    r31 = -np.sin(ax)
    r32 = np.sin(ax)*np.cos(ay)
    r33 = np.cos(ax)*np.cos(ay)
    
    R = np.zeros([3, 3], dtype=float)
    
    R[0, 0] = r11
    R[0, 1] = r12
    R[0, 2] = r13
    
    R[1, 0] = r21
    R[1, 1] = r22
    R[1, 2] = r23
    
    R[2, 0] = r31
    R[2, 1] = r32
    R[2, 2] = r33
    
    return R

def generate_seed(_nca_params):
    PAD_SIZE = _nca_params['_SIZE_']+(2*_nca_params['_PAD_'])
    if _nca_params['_USE_SPHERE_SEED_']:
        seed_ten = seed_3d(_size=PAD_SIZE, _channels=_nca_params['_CHANNELS_'], _points=_nca_params['_SEED_POINTS_'], _radius=_nca_params['_SEED_DIST_']).unsqueeze(0)
    else:
        seed_dic = _nca_params['_SEED_DIC_']
        seed_ten = custom_seed(_size=PAD_SIZE, _channels=_nca_params['_CHANNELS_'], _dist=_nca_params['_SEED_DIST_'], _hidden_info=_nca_params['_SEED_HID_INFO_'],
                                    _center=seed_dic['center'], 
                                    _plus_x=seed_dic['plus_x'], _minus_x=seed_dic['minus_x'],
                                    _plus_y=seed_dic['plus_y'], _minus_y=seed_dic['minus_y'],
                                    _plus_z=seed_dic['plus_z'], _minus_z=seed_dic['minus_z']).unsqueeze(0)
    return seed_ten

def load_vox_as_tensor(_nca_params):
    target_vox = _nca_params['_TARGET_VOX_']
    if target_vox.endswith('vox'):
        target = Vox().load_from_file(target_vox)
        target_ten = target.tensor()
    elif target_vox.endswith('npy'):
        with open(target_vox, 'rb') as f:
            target_ten = torch.from_numpy(np.load(f))
    
    pad = _nca_params['_PAD_']
    target_ten = func.pad(target_ten, (pad, pad, pad, pad, pad, pad), 'constant')
    target_ten = target_ten.clone().repeat(_nca_params['_BATCH_SIZE_'], 1, 1, 1, 1)
    
def generate_pool(_nca_params, _seed_ten, _isotype):
    pool_size = _nca_params['_POOL_SIZE_']
    pad = _nca_params['PAD_SIZE']
    with torch.no_grad():
        pool = _seed_ten.clone().repeat(pool_size, 1, 1, 1, 1)
        # * randomize channel(s)
        if _isotype == 1:
            for j in range(pool_size):
                pool[j, -1:] = torch.rand(pad, pad, pad)*np.pi*2.0
        elif _isotype == 3:
            for j in range(pool_size):
                pool[j, -1:] = torch.rand(pad, pad, pad)*np.pi*2.0
                pool[j, -2:-1] = torch.rand(pad, pad, pad)*np.pi*2.0
                pool[j, -3:-2] = torch.rand(pad, pad, pad)*np.pi*2.0

    return pool