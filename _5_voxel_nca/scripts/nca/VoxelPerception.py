import torch
import torch.nn.functional as func
from scripts.nca import VoxelUtil as util
from scipy.spatial.transform import Rotation

# 3D filters
X_SOBEL_KERN = torch.tensor([
   [[1., 2., 1.], 
    [0., 0., 0.], 
    [-1., -2., -1.]],
   
   [[2., 4., 2.], 
    [0., 0., 0.], 
    [-2., -4., -2.]],
   
   [[1., 2., 1.], 
    [0., 0., 0.], 
    [-1., -2., -1.]]])
Y_SOBEL_KERN = torch.tensor([
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
    
    def yaw_isotropic_perception(self, _x):
        # * separate states and angle channels
        states, angle = _x[:, :-1], _x[:, -1:]
        # * calculate gx and gy
        gx = self.per_channel_conv3d(states, X_SOBEL_KERN[None, :])
        gy = self.per_channel_conv3d(states, Y_SOBEL_KERN[None, :])
        # * compute px and py 
        _cos, _sin = angle.cos(), angle.sin()
        px = (gx*_cos)+(gy*_sin)
        py = (gy*_cos)-(gx*_sin)
        # * calculate gz and lap
        gz = self.per_channel_conv3d(states, Z_SOBEL_KERN[None, :])
        lap = self.per_channel_conv3d(states, LAP_KERN[None, :])
        return torch.cat([_x, px, py, gz, lap], 1)
    
    def quaternion_perception(self, _x):
        # * separate states and angle channels
        states, ax, ay, az = _x[:, :-3], _x[:, -1:], _x[:, -2:-1], _x[:, -3:-2]
        # print (f'states.shape: {states.shape}')
        # print (f'ax.shape: {ax.shape}')
        # print (f'ay.shape: {ay.shape}')
        # print (f'az.shape: {az.shape}')
        # * per channel convolutions
        px = self.per_channel_conv3d(states, X_SOBEL_KERN[None, :])
        py = self.per_channel_conv3d(states, Y_SOBEL_KERN[None, :])
        pz = self.per_channel_conv3d(states, Z_SOBEL_KERN[None, :])
        lap = self.per_channel_conv3d(states, LAP_KERN[None, :])
        # * get quat values
        quat = util.euler_to_quaternion(ax, ay, az)
        # print (f'quat.shape: {quat.shape}')
    
        perception_tensors = [px, py, pz]
        
        # Initialize empty output tensors for each direction (px, py, pz)
        batch_size, hidden_channels, x, y, z = px.shape
        rotated_tensors = [
            torch.zeros((batch_size, hidden_channels, x, y, z)),
            torch.zeros((batch_size, hidden_channels, x, y, z)),
            torch.zeros((batch_size, hidden_channels, x, y, z))
        ]

        # Iterate through each cell in the volume and apply the rotation
        for i in range(x):
            for j in range(y):
                for k in range(z):
                    cell_quaternion = quat[:, :, i, j, k]
                    cell_perception = torch.zeros((batch_size, 4, hidden_channels))
                    cell_rotated = torch.zeros((batch_size, 4, hidden_channels))

                    # Iterate through each direction (px, py, pz)
                    for t in range(3):
                        # Extract the perception tensors for this direction
                        cell_perception[:, t+1] = perception_tensors[t][:, :, i, j, k]
                        
                    # print (f'cell_quaternion.shape: {cell_quaternion.shape}')
                    # print (f'cell_perception.shape: {cell_perception.shape}')
                    
                    for t in range(hidden_channels):
                        cell_rotated[:, :, t] = cell_quaternion * cell_perception[:, :, t] * torch.conj(cell_quaternion)

                    # print (f'cell_rotated[:, 0].shape: {cell_rotated[:, 0].shape}')
                    # print (f'rotated_tensors[0][:, :, i, j, k].shape: {rotated_tensors[0][:, :, i, j, k].shape}')
                    
                    for t in range(3):
                        rotated_tensors[t][:, :, i, j, k] = cell_rotated[:, t+1]
                        
        return torch.cat([_x, rotated_tensors[0], rotated_tensors[1], rotated_tensors[2], lap], 1)
        
        
    perception = {
        'ANISOTROPIC': anisotropic_perception,
        'YAW_ISO': yaw_isotropic_perception,
        'QUATERNION': quaternion_perception,
    }