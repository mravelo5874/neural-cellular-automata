import torch
import torch.nn.functional as func

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
        # * per channel convolutions
        gx = self.per_channel_conv3d(states, X_SOBEL_KERN[None, :])
        gy = self.per_channel_conv3d(states, Y_SOBEL_KERN[None, :])
        gz = self.per_channel_conv3d(states, Z_SOBEL_KERN[None, :])
        lap = self.per_channel_conv3d(states, LAP_KERN[None, :])
        print ('gx.shape:',gx.shape)
        print ('gy.shape:',gy.shape)
        # * compute px and py 
        _cos, _sin = angle.cos(), angle.sin()
        px = torch.empty(gx.shape, dtype=torch.float32)
        py = torch.empty(gx.shape, dtype=torch.float32)
        print ('px.shape:',px.shape)
        print ('py.shape:',py.shape)
        for i in range(3):
            i_x = gx[:, :, :, i]
            i_y = gy[:, :, :, i]
            print ('*i_x.shape:',i_x.shape)
            print ('*gx[:, :, :, i].shape:',gx[:, :, :, i].shape)
            print ('px[:, :, :, i].shape:',px[:, :, :, i].shape)
            print ('py[:, :, :, i].shape:',py[:, :, :, i].shape)
            px[:, :, :, i] = (i_x*_cos)+(i_y*_sin)
            py[:, :, :, i] = (i_y*_cos)-(i_x*_sin)
        return torch.cat([_x, px, py, gz, lap], 1)
        
    perception = {
        'ANISOTROPIC': anisotropic_perception,
        'YAW_ISO': yaw_isotropic_perception,
    }