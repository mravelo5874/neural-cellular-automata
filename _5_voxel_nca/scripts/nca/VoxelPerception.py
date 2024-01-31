import torch
import torch.nn.functional as func
import numpy as np
import scipy.spatial.transform as sci
from pytorch3d.transforms import quaternion_multiply 

from enum import Enum
from scripts.nca import VoxelUtil as voxutil
from scripts.vox.Vox import Vox

class Perception(int, Enum):
    ANISOTROPIC: int = 0                                      
    YAW_ISO: int = 1                                       
    QUATERNION: int = 2
    FAST_QUAT: int = 3
    EULER: int = 4
    YAW_ISO_V2: int = 5
    YAW_ISO_V3: int = 6

# 3D filters
Y_SOBEL_2D_KERN = torch.tensor([
   [[0., 0., 0.], 
    [0., 0., 0.], 
    [0., 0., 0.]],
   
   [[1., 2., 1.], 
    [0., 0., 0.], 
    [-1., -2., -1.]],
   
   [[0., 0., 0.], 
    [0., 0., 0.], 
    [0., 0., 0.]]])

X_SOBEL_2D_KERN = torch.tensor([
   [[0., 0., 0.], 
    [1., 2., 1.], 
    [0., 0., 0.]],
   
   [[0., 0., 0.], 
    [0., 0., 0.], 
    [0., 0., 0.]],
   
   [[0., 0., 0.], 
    [-1., -2., -1.],
    [0., 0., 0.]]])

Y_SOBEL_2D_KERN_v2 = torch.tensor([
   [[0., 1., 0.], 
    [0., 0., 0.], 
    [0., -1., 0.]],
   
   [[0., 2., 0.], 
    [0., 0., 0.], 
    [0., -2., 0.]],
   
   [[0., 1., 0.], 
    [0., 0., 0.], 
    [0., -1., 0.]]])

X_SOBEL_2D_KERN_v2 = torch.tensor([
   [[0., 1., 0.], 
    [0., 2., 0.], 
    [0., 1., 0.]],
   
   [[0., 0., 0.], 
    [0., 0., 0.], 
    [0., 0., 0.]],
   
   [[0., -1., 0.], 
    [0., -2., 0.], 
    [0., -1., 0.]]])

Z_SOBEL_2D_KERN = torch.tensor([
   [[0., 0., 0.], 
    [0., 0., 0.], 
    [0., 0., 0.]],
   
   [[-1., 0., 1.], 
    [-2., 0., 2.], 
    [-1., 0., 1.]],
   
   [[0., 0., 0.], 
    [0., 0., 0.], 
    [0., 0., 0.]]])

Y_SOBEL_KERN = torch.tensor([
   [[1., 2., 1.], 
    [0., 0., 0.], 
    [-1., -2., -1.]],
   
   [[2., 4., 2.], 
    [0., 0., 0.], 
    [-2., -4., -2.]],
   
   [[1., 2., 1.], 
    [0., 0., 0.], 
    [-1., -2., -1.]]])

X_SOBEL_KERN = torch.tensor([
   [[1., 2., 1.], 
    [2., 4., 2.], 
    [1., 2., 1.]],
   
   [[0., 0., 0.], 
    [0., 0., 0.], 
    [0., 0., 0.]],
   
   [[-1., -2., -1.], 
    [-2., -4., -2.], 
    [-1., -2., -1.]]])

Z_SOBEL_KERN = torch.tensor([
   [[-1., 0., 1.], 
    [-2., 0., 2.], 
    [-1., 0., 1.]],
   
   [[-2., 0., 2.], 
    [-4., 0., 4.], 
    [-2., 0., 2.]],
   
   [[-1., 0., 1.], 
    [-2., 0., 2.], 
    [-1., 0., 1.]]])

LAP_KERN = torch.tensor([
   [[2., 3., 2.], 
    [3., 6., 3.], 
    [2., 3., 2.]],
   
   [[3., 6., 3.], 
    [6.,-88., 6.], 
    [3., 6., 3.]],
   
   [[2., 3., 2.], 
    [3., 6., 3.], 
    [2., 3., 2.]]])/26.0

class VoxelPerception():
    def __init__(self, _device='cuda'):
        super().__init__()
        self.device = _device
        
    # * performs a convolution per filter per channel
    def per_channel_conv3d(self, _x, _filters):
        batch_size, channels, height, width, depth = _x.shape
        # * reshape x to make per-channel convolution possible + pad 1 on each side
        y = _x.reshape(batch_size*channels, 1, height, width, depth)
        y = func.pad(y, (1, 1, 1, 1, 1, 1), 'constant')
        # * perform per-channel convolutions
        _filters = _filters.to(self.device)
        y = func.conv3d(y, _filters[:, None])
        y = y.reshape(batch_size, -1, height, width, depth)
        return y

    def anisotropic_perception(self, _x):
        _x = _x.to(self.device)
        # * per channel convolutions
        gx = self.per_channel_conv3d(_x, X_SOBEL_KERN[None, :])
        gy = self.per_channel_conv3d(_x, Y_SOBEL_KERN[None, :])
        gz = self.per_channel_conv3d(_x, Z_SOBEL_KERN[None, :])
        lap = self.per_channel_conv3d(_x, LAP_KERN[None, :])
        return torch.cat([_x, gx, gy, gz, lap], 1)
    
    # * issues:
    # *     - found an 'if' that needed to be an 'elif'
    # *     - final perception concatinates all _x when should just be just states
    # *     - old 3d sobel filters gathered info in z direction (could still work?)
    def yaw_isotropic_perception(self, _x, _c=None):
        # * separate states and angle channels
        states, angle = _x[:, :-1], _x[:, -1:]
        
        if _c != None:
            c_states, c_angle = _c[:, :-1], _c[:, -1:]
            
            c_states_clone = c_states.detach().clone()
            c_states_clone = torch.rot90(c_states_clone, 1, (3, 2))
            
            dif = torch.abs(states - c_states_clone)
            res = torch.all(dif < 0.0001)
            print (f'states comp: {res}')
            
            c_angle_clone = c_angle.detach().clone()
            c_angle_clone = torch.sub(c_angle_clone, (np.pi/2))
            c_angle_clone = torch.rot90(c_angle_clone, 1, (3, 2))
            
            dif = torch.abs(angle - c_angle_clone)
            res = torch.all(dif < 0.0001)
            print (f'angle comp: {res}')
        
        # * calculate gx and gy
        gx = self.per_channel_conv3d(states, X_SOBEL_KERN[None, :])
        gy = self.per_channel_conv3d(states, Y_SOBEL_KERN[None, :])
        
        if _c != None:
            c_gx = self.per_channel_conv3d(c_states, X_SOBEL_KERN[None, :])
            c_gy = self.per_channel_conv3d(c_states, Y_SOBEL_KERN[None, :])
            
            c_gx_clone = c_gx.detach().clone()
            c_gx_clone = torch.rot90(c_gx_clone, 1, (3, 2))
            
            # print ('* rendering gx...')
            # Vox().load_from_tensor(gx).render(_show_grid=True)
            # print ('* rendering c_gx_clone...')
            # Vox().load_from_tensor(c_gx_clone).render(_show_grid=True)
            
            dif = torch.abs(gx - c_gx_clone)
            res = torch.all(dif < 0.0001)
            print (f'gx comp: {res}')
            
            c_gy_clone = c_gy.detach().clone()
            c_gy_clone = torch.rot90(c_gy_clone, 1, (3, 2))
            
            # print ('* rendering gy...')
            # Vox().load_from_tensor(gy).render(_show_grid=True)
            # print ('* rendering c_gy_clone...')
            # Vox().load_from_tensor(c_gy_clone).render(_show_grid=True)
            
            dif = torch.abs(gy - c_gy_clone)
            res = torch.all(dif < 0.0001)
            print (f'gy comp: {res}')
            
        # * compute px and py 
        _cos, _sin = angle.cos(), angle.sin()
        
        if _c != None:
            c_cos, c_sin = c_angle.cos(), c_angle.sin()
        
        px = (gx*_cos)+(gy*_sin)
        py = (gy*_cos)-(gx*_sin)
        
        if _c != None:
            c_px = (c_gx*c_cos)+(c_gy*c_sin)
            c_py = (c_gy*c_cos)-(c_gx*c_sin)
            
            c_px_clone = c_px.detach().clone()
            c_px_clone = torch.rot90(c_px_clone, 1, (3, 2))
            
            # print ('* rendering px...')
            # Vox().load_from_tensor(px).render(_show_grid=True)
            # print ('* rendering c_px_clone...')
            # Vox().load_from_tensor(c_px_clone).render(_show_grid=True)
            
            dif = torch.abs(px - c_px_clone)
            res = torch.all(dif < 0.0001)
            print (f'px comp: {res}')
            
            c_py_clone = c_py.detach().clone()
            c_py_clone = torch.rot90(c_py_clone, 1, (3, 2))
            
            # print ('* rendering py...')
            # Vox().load_from_tensor(py).render(_show_grid=True)
            # print ('* rendering c_py_clone...')
            # Vox().load_from_tensor(c_py_clone).render(_show_grid=True)
            
            dif = torch.abs(py - c_py_clone)
            res = torch.all(dif < 0.0001)
            print (f'py comp: {res}')
        
        # * calculate gz and lap
        gz = self.per_channel_conv3d(states, Z_SOBEL_KERN[None, :])
        lap = self.per_channel_conv3d(states, LAP_KERN[None, :])
        
        if _c != None:
            c_gz = self.per_channel_conv3d(c_states, Z_SOBEL_KERN[None, :])
            c_lap = self.per_channel_conv3d(c_states, LAP_KERN[None, :])
            
            c_gz_clone = c_gz.detach().clone()
            c_gz_clone = torch.rot90(c_gz_clone, 1, (3, 2))
            
            # print ('* rendering gz...')
            # Vox().load_from_tensor(gz).render(_show_grid=True)
            # print ('* rendering c_gz_clone...')
            # Vox().load_from_tensor(c_gz_clone).render(_show_grid=True)
            
            dif = torch.abs(gz - c_gz_clone)
            res = torch.all(dif < 0.0001)
            print (f'gz comp: {res}')
            
            c_lap_clone = c_lap.detach().clone()
            c_lap_clone = torch.rot90(c_lap_clone, 1, (3, 2))
            
            # print ('* rendering lap...')
            # Vox().load_from_tensor(lap).render(_show_grid=True)
            # print ('* rendering c_lap_clone...')
            # Vox().load_from_tensor(c_lap_clone).render(_show_grid=True)
            
            dif = torch.abs(lap - c_lap_clone)
            res = torch.all(dif < 0.0001)
            print (f'lap comp: {res}')
            
        return torch.cat([_x, px, py, gz, lap], 1)
    
    
    
    
    
    def yaw_isotropic_v2_perception(self, _x, _c=None, _iso_type=5):
        # * separate states and angle channels
        states, angle = _x[:, :-1], _x[:, -1:]
        
        if _c != None:
            c_clone = _c.detach().clone()
            c_clone = torch.rot90(c_clone, 1, (3, 2))
            
            # * rotate istropic channel(s)
            if _iso_type == 1:
                c_clone[:, -1:] = (torch.sub(c_clone[:, -1:], (np.pi/2))) % (np.pi*2)
            elif _iso_type == 3:
                c_clone[:, -1:] = (torch.sub(c_clone[:, -1:], (np.pi/2))) % (np.pi*2)
                c_clone[:, -2:-1] = (torch.sub(c_clone[:, -2:-1], (np.pi/2))) % (np.pi*2)
                c_clone[:, -3:-2] = (torch.sub(c_clone[:, -3:-2], (np.pi/2))) % (np.pi*2)
            
            dif = torch.abs(_x - c_clone)
            res = torch.all(dif < 0.0001)
            print (f'x/c comp: {res}')
            
            # print ('* rendering x...')
            # Vox().load_from_tensor(_x).render(_show_grid=True)
            # print ('* rendering c_clone...')
            # Vox().load_from_tensor(c_clone).render(_show_grid=True)
            
            c_states, c_angle = _c[:, :-1], _c[:, -1:]
            
            c_states_clone = c_states.detach().clone()
            c_states_clone = torch.rot90(c_states_clone, 1, (3, 2))
            
            dif = torch.abs(states - c_states_clone)
            res = torch.all(dif < 0.0001)
            print (f'states comp: {res}')
            
            c_angle_clone = c_angle.detach().clone()
            c_angle_clone = torch.sub(c_angle_clone, (np.pi/2))
            c_angle_clone = torch.rot90(c_angle_clone, 1, (3, 2))
            
            dif = torch.abs(angle - c_angle_clone)
            res = torch.all(dif < 0.0001)
            print (f'angle comp: {res}')
        
        # * calculate gx and gy
        gx = self.per_channel_conv3d(states, X_SOBEL_2D_KERN_v2[None, :])
        gy = self.per_channel_conv3d(states, Y_SOBEL_2D_KERN_v2[None, :])
        
        if _c != None:
            c_gx = self.per_channel_conv3d(c_states, X_SOBEL_2D_KERN_v2[None, :])
            c_gy = self.per_channel_conv3d(c_states, Y_SOBEL_2D_KERN_v2[None, :])
            
            c_gx_clone = c_gx.detach().clone()
            c_gx_clone = torch.rot90(c_gx_clone, 1, (3, 2))
            
            # print ('* rendering gx...')
            # Vox().load_from_tensor(gx).render(_show_grid=True)
            # print ('* rendering c_gx_clone...')
            # Vox().load_from_tensor(c_gx_clone).render(_show_grid=True)
            
            dif = torch.abs(gx - c_gx_clone)
            res = torch.all(dif < 0.0001)
            print (f'gx comp: {res}')
            
            c_gy_clone = c_gy.detach().clone()
            c_gy_clone = torch.rot90(c_gy_clone, 1, (3, 2))
            
            # print ('* rendering gy...')
            # Vox().load_from_tensor(gy).render(_show_grid=True)
            # print ('* rendering c_gy_clone...')
            # Vox().load_from_tensor(c_gy_clone).render(_show_grid=True)
            
            dif = torch.abs(gy - c_gy_clone)
            res = torch.all(dif < 0.0001)
            print (f'gy comp: {res}')
           
        # * compute px and py 
        _cos, _sin = angle.cos(), angle.sin()

        if _c != None:
            c_cos, c_sin = c_angle.cos(), c_angle.sin()
   
        px = (gy*_cos)-(gx*_sin)
        py = (gy*_sin)+(gx*_cos)
        
        if _c != None:
            c_px = (c_gy*c_cos)-(c_gx*c_sin)
            c_py = (c_gy*c_sin)+(c_gx*c_cos)
            
            c_px_clone = c_px.detach().clone()
            c_px_clone = torch.rot90(c_px_clone, 1, (3, 2))
            
            # print ('* rendering px...')
            # Vox().load_from_tensor(px).render(_show_grid=True)
            # print ('* rendering c_px_clone...')
            # Vox().load_from_tensor(c_px_clone).render(_show_grid=True)
            
            dif = torch.abs(px - c_px_clone)
            res = torch.all(dif < 0.0001)
            print (f'px comp: {res}')
            
            c_py_clone = c_py.detach().clone()
            c_py_clone = torch.rot90(c_py_clone, 1, (3, 2))
            
            # print ('* rendering py...')
            # Vox().load_from_tensor(py).render(_show_grid=True)
            # print ('* rendering c_py_clone...')
            # Vox().load_from_tensor(c_py_clone).render(_show_grid=True)
            
            dif = torch.abs(py - c_py_clone)
            res = torch.all(dif < 0.0001)
            print (f'py comp: {res}')
        
        # * calculate gz and lap
        gz = self.per_channel_conv3d(states, Z_SOBEL_KERN[None, :])
        lap = self.per_channel_conv3d(states, LAP_KERN[None, :])
        
        if _c != None:
            c_gz = self.per_channel_conv3d(c_states, Z_SOBEL_KERN[None, :])
            c_lap = self.per_channel_conv3d(c_states, LAP_KERN[None, :])
            
            c_gz_clone = c_gz.detach().clone()
            c_gz_clone = torch.rot90(c_gz_clone, 1, (3, 2))
            
            # print ('* rendering gz...')
            # Vox().load_from_tensor(gz).render(_show_grid=True)
            # print ('* rendering c_gz_clone...')
            # Vox().load_from_tensor(c_gz_clone).render(_show_grid=True)
            
            dif = torch.abs(gz - c_gz_clone)
            res = torch.all(dif < 0.0001)
            print (f'gz comp: {res}')
            
            c_lap_clone = c_lap.detach().clone()
            c_lap_clone = torch.rot90(c_lap_clone, 1, (3, 2))
            
            # print ('* rendering lap...')
            # Vox().load_from_tensor(lap).render(_show_grid=True)
            # print ('* rendering c_lap_clone...')
            # Vox().load_from_tensor(c_lap_clone).render(_show_grid=True)
            
            dif = torch.abs(lap - c_lap_clone)
            res = torch.all(dif < 0.0001)
            print (f'lap comp: {res}')
        
        return torch.cat([states, px, py, gz, lap], 1)
    
    def yaw_isotropic_v3_perception(self, _x, _c=None, _iso_type=5):
        # * separate states and angle channels
        states, angle = _x[:, :-1], _x[:, -1:]
        
        if _c != None:
            c_clone = _c.detach().clone()
            c_clone = torch.rot90(c_clone, 1, (3, 2))
            
            # * rotate istropic channel(s)
            if _iso_type == 1:
                c_clone[:, -1:] = (torch.sub(c_clone[:, -1:], (np.pi/2))) % (np.pi*2)
            elif _iso_type == 3:
                c_clone[:, -1:] = (torch.sub(c_clone[:, -1:], (np.pi/2))) % (np.pi*2)
                c_clone[:, -2:-1] = (torch.sub(c_clone[:, -2:-1], (np.pi/2))) % (np.pi*2)
                c_clone[:, -3:-2] = (torch.sub(c_clone[:, -3:-2], (np.pi/2))) % (np.pi*2)
            
            dif = torch.abs(_x - c_clone)
            res = torch.all(dif < 0.0001)
            print (f'x/c comp: {res}')
            
            # print ('* rendering x...')
            # Vox().load_from_tensor(_x).render(_show_grid=True)
            # print ('* rendering c_clone...')
            # Vox().load_from_tensor(c_clone).render(_show_grid=True)
            
            c_states, c_angle = _c[:, :-1], _c[:, -1:]
            
            c_states_clone = c_states.detach().clone()
            c_states_clone = torch.rot90(c_states_clone, 1, (3, 2))
            
            dif = torch.abs(states - c_states_clone)
            res = torch.all(dif < 0.0001)
            print (f'states comp: {res}')
            
            c_angle_clone = c_angle.detach().clone()
            c_angle_clone = torch.sub(c_angle_clone, (np.pi/2))
            c_angle_clone = torch.rot90(c_angle_clone, 1, (3, 2))
            
            dif = torch.abs(angle - c_angle_clone)
            res = torch.all(dif < 0.0001)
            print (f'angle comp: {res}')
        
        # * calculate gx and gy
        gx = self.per_channel_conv3d(states, X_SOBEL_KERN[None, :])
        gy = self.per_channel_conv3d(states, Y_SOBEL_KERN[None, :])
        
        if _c != None:
            c_gx = self.per_channel_conv3d(c_states, X_SOBEL_KERN[None, :])
            c_gy = self.per_channel_conv3d(c_states, Y_SOBEL_KERN[None, :])
            
            c_gx_clone = c_gx.detach().clone()
            c_gx_clone = torch.rot90(c_gx_clone, 1, (3, 2))
            
            # print ('* rendering gx...')
            # Vox().load_from_tensor(gx).render(_show_grid=True)
            # print ('* rendering c_gx_clone...')
            # Vox().load_from_tensor(c_gx_clone).render(_show_grid=True)
            
            dif = torch.abs(gx - c_gx_clone)
            res = torch.all(dif < 0.0001)
            print (f'gx comp: {res}')
            
            c_gy_clone = c_gy.detach().clone()
            c_gy_clone = torch.rot90(c_gy_clone, 1, (3, 2))
            
            # print ('* rendering gy...')
            # Vox().load_from_tensor(gy).render(_show_grid=True)
            # print ('* rendering c_gy_clone...')
            # Vox().load_from_tensor(c_gy_clone).render(_show_grid=True)
            
            dif = torch.abs(gy - c_gy_clone)
            res = torch.all(dif < 0.0001)
            print (f'gy comp: {res}')
           
        # * compute px and py 
        _cos, _sin = angle.cos(), angle.sin()

        if _c != None:
            c_cos, c_sin = c_angle.cos(), c_angle.sin()
   
        px = (gy*_cos)-(gx*_sin)
        py = (gy*_sin)+(gx*_cos)
        
        if _c != None:
            c_px = (c_gy*c_cos)-(c_gx*c_sin)
            c_py = (c_gy*c_sin)+(c_gx*c_cos)
            
            c_px_clone = c_px.detach().clone()
            c_px_clone = torch.rot90(c_px_clone, 1, (3, 2))
            
            # print ('* rendering px...')
            # Vox().load_from_tensor(px).render(_show_grid=True)
            # print ('* rendering c_px_clone...')
            # Vox().load_from_tensor(c_px_clone).render(_show_grid=True)
            
            dif = torch.abs(px - c_px_clone)
            res = torch.all(dif < 0.0001)
            print (f'px comp: {res}')
            
            c_py_clone = c_py.detach().clone()
            c_py_clone = torch.rot90(c_py_clone, 1, (3, 2))
            
            # print ('* rendering py...')
            # Vox().load_from_tensor(py).render(_show_grid=True)
            # print ('* rendering c_py_clone...')
            # Vox().load_from_tensor(c_py_clone).render(_show_grid=True)
            
            dif = torch.abs(py - c_py_clone)
            res = torch.all(dif < 0.0001)
            print (f'py comp: {res}')
        
        # * calculate gz and lap
        gz = self.per_channel_conv3d(states, Z_SOBEL_KERN[None, :])
        lap = self.per_channel_conv3d(states, LAP_KERN[None, :])
        
        if _c != None:
            c_gz = self.per_channel_conv3d(c_states, Z_SOBEL_KERN[None, :])
            c_lap = self.per_channel_conv3d(c_states, LAP_KERN[None, :])
            
            c_gz_clone = c_gz.detach().clone()
            c_gz_clone = torch.rot90(c_gz_clone, 1, (3, 2))
            
            # print ('* rendering gz...')
            # Vox().load_from_tensor(gz).render(_show_grid=True)
            # print ('* rendering c_gz_clone...')
            # Vox().load_from_tensor(c_gz_clone).render(_show_grid=True)
            
            dif = torch.abs(gz - c_gz_clone)
            res = torch.all(dif < 0.0001)
            print (f'gz comp: {res}')
            
            c_lap_clone = c_lap.detach().clone()
            c_lap_clone = torch.rot90(c_lap_clone, 1, (3, 2))
            
            # print ('* rendering lap...')
            # Vox().load_from_tensor(lap).render(_show_grid=True)
            # print ('* rendering c_lap_clone...')
            # Vox().load_from_tensor(c_lap_clone).render(_show_grid=True)
            
            dif = torch.abs(lap - c_lap_clone)
            res = torch.all(dif < 0.0001)
            print (f'lap comp: {res}')
        
        return torch.cat([states, px, py, gz, lap], 1)
    
    def quaternion_perception(self, _x):
        # * separate states and angle channels
        states, ax, ay, az = _x[:, :-3], _x[:, -1:], _x[:, -2:-1], _x[:, -3:-2]

        # * per channel convolutions
        px = self.per_channel_conv3d(states, X_SOBEL_KERN[None, :])
        py = self.per_channel_conv3d(states, Y_SOBEL_KERN[None, :])
        pz = self.per_channel_conv3d(states, Z_SOBEL_KERN[None, :])
        lap = self.per_channel_conv3d(states, LAP_KERN[None, :])
        
        # * combine perception tensors
        px = px[:, None, ...]
        py = py[:, None, ...]
        pz = pz[:, None, ...]
        p0 = torch.zeros_like(px)
        
        # * get quat values
        bs, a, sx, sy, sz = ax.shape
        ax = ax.reshape([bs, a, sx*sy*sz])
        ay = ay.reshape([bs, a, sx*sy*sz])
        az = az.reshape([bs, a, sx*sy*sz])
        quats = voxutil.euler_to_quaternion(ax, ay, az)
        
        # * concat and prepare for rotation
        pxyz = torch.cat([p0, px, py, pz], 1)
        bs, p4, hc, sx, sy, sz = pxyz.shape
        pxyz = pxyz.reshape([bs, p4, hc, sx*sy*sz])
        conj = torch.conj(quats)
        
        # * permute to get [4] on last dim
        # * needed for: quaternion_multiply()
        pxyz = pxyz.permute([0, 3, 2, 1])
        conj = conj.permute([0, 2, 1])
        quats = quats.permute([0, 2, 1])
        
        # * rotate perception tensors
        rxyz = torch.zeros_like(pxyz)
        for t in range(hc):
            res = quaternion_multiply(quats, pxyz[:, :, t])
            res = quaternion_multiply(res, conj)
            rxyz[:, :, t] = res
            
        rxyz = rxyz.permute([0, 3, 2, 1])
        rxyz = rxyz.reshape([bs, p4, hc, sx, sy, sz])
        
        # * extract rotated perception tensors
        rx = rxyz[:, 1]
        ry = rxyz[:, 2]
        rz = rxyz[:, 3]
        return torch.cat([states, rx, ry, rz, lap], 1)
    
    def fast_quat_perception(self, _x):
        # * separate states and angle channels
        states, ax, ay, az = _x[:, :-3], _x[:, -1:], _x[:, -2:-1], _x[:, -3:-2]
        
        # * per channel convolutions
        px = self.per_channel_conv3d(states, X_SOBEL_KERN[None, :])
        py = self.per_channel_conv3d(states, Y_SOBEL_KERN[None, :])
        pz = self.per_channel_conv3d(states, Z_SOBEL_KERN[None, :])
        lap = self.per_channel_conv3d(states, LAP_KERN[None, :])
        
        # * combine perception tensors
        px = px[None, ...]
        py = py[None, ...]
        pz = pz[None, ...]
        pxyz = torch.cat([px, py, pz], 0)
        p3, bs, hc, sx, sy, sz = pxyz.shape
        pxyz = torch.permute(pxyz, (2, 1, 3, 4, 5, 0))
        pxyz = pxyz.reshape([hc, bs*sx*sy*sz, p3])
        
        # * get quat values
        axyz = torch.cat([ax, ay, az], 1)
        bs, a, sx, sy, sz = axyz.shape
        axyz = torch.permute(axyz, (0, 2, 3, 4, 1))
        axyz = axyz.reshape([sx*sy*sz*bs, a])
        #quat = T.axis_angle_to_quaternion(axyz)
        
        # * apply quat rotations
        rxyz = torch.zeros_like(pxyz)
        for t in range(hc):
            #rxyz[t] = T.quaternion_apply(quat, pxyz[t])
            rxyz[t] = pxyz[t]
        rxyz = torch.permute(rxyz, (2, 0, 1))
        rxyz = rxyz.reshape([bs, p3, hc, sx, sy, sz])

        # * extract rotated perception tensors
        rx = rxyz[:, 0]
        ry = rxyz[:, 1]
        rz = rxyz[:, 2]
        return torch.cat([_x, rx, ry, rz, lap], 1)
    
    def euler_perception(self, _x):
        # * separate states and angle channels
        states, ax, ay, az = _x[:, :-3], _x[:, -1:], _x[:, -2:-1], _x[:, -3:-2]

        # * per channel convolutions
        px = self.per_channel_conv3d(states, X_SOBEL_KERN[None, :])
        py = self.per_channel_conv3d(states, Y_SOBEL_KERN[None, :])
        pz = self.per_channel_conv3d(states, Z_SOBEL_KERN[None, :])
        lap = self.per_channel_conv3d(states, LAP_KERN[None, :])
        
        # * combine perception tensors
        px = px[None, ...]
        py = py[None, ...]
        pz = pz[None, ...]
        pxyz = torch.cat([px, py, pz], 0)
        p3, bs, hc, sx, sy, sz = pxyz.shape
        pxyz = torch.permute(pxyz, (2, 1, 3, 4, 5, 0))
        pxyz = pxyz.reshape([hc, bs*sx*sy*sz, p3])
 
        # * get quat values
        axyz = torch.cat([ax, ay, az], 1)
        bs, a, sx, sy, sz = axyz.shape
        axyz = torch.permute(axyz, (0, 2, 3, 4, 1))
        axyz = axyz.reshape([sx*sy*sz*bs, a])

        # * perform rotations
        rots = sci.Rotation.from_euler('xyz', axyz.cpu().detach().numpy(), degrees=False)
        rxyz = np.zeros_like(pxyz.cpu().detach().numpy())
        for t in range(hc):
            rxyz[t] = rots.apply(pxyz[t].cpu().detach().numpy())
        rxyz = torch.tensor(rxyz)
        rxyz = torch.permute(rxyz, (2, 0, 1))
        rxyz = rxyz.reshape([bs, p3, hc, sx, sy, sz])

        # * extract rotated perception tensors
        rx = rxyz[:, 0]
        ry = rxyz[:, 1]
        rz = rxyz[:, 2]
        return torch.cat([_x, rx, ry, rz, lap], 1)
        
    perception = {
        Perception.ANISOTROPIC: anisotropic_perception,
        Perception.YAW_ISO: yaw_isotropic_perception,
        Perception.QUATERNION: quaternion_perception,
        Perception.FAST_QUAT: fast_quat_perception,
        Perception.EULER: euler_perception,
        Perception.YAW_ISO_V2: yaw_isotropic_v2_perception,
        Perception.YAW_ISO_V3: yaw_isotropic_v3_perception,
    }