import torch
import torch.nn.functional as func
from VoxelPerception import perception
from numpy import pi as PI

def create_seed(_size=16, _channels=16, _dist=5, _points=4):
    x = torch.zeros([_size, _size, _size, _channels])
    half = _size//2
    # * red
    if _points > 0:
        x[half, half, half, 3:_channels] = 1.0
        x[half, half, half, 0] = 1.0
    # * green
    if _points > 1:
        x[half, half+_dist, half, 3:_channels] = 1.0
        x[half, half+_dist, half, 1] = 1.0
    # * blue
    if _points > 2:
        x[half+_dist, half, half, 3:_channels] = 1.0
        x[half+_dist, half, half, 2] = 1.0
    # * yellow
    if _points > 3:
        x[half, half, half+_dist, 3:_channels] = 1.0
        x[half, half, half+_dist, 0:2] = 1.0
    # * magenta
    if _points > 4:
        x[half, half-_dist, half, 3:_channels] = 1.0
        x[half, half-_dist, half, 0] = 1.0
        x[half, half-_dist, half, 2] = 1.0
    # * cyan
    if _points > 5:
        x[half-_dist, half, half, 3:_channels] = 1.0
        x[half-_dist, half, half, 1:3] = 1.0
    return x
    
class VoxelNCA(torch.nn.Module):
    def __init__(self, _channels=16, _hidden=128, _device='cuda', _model_type='STEERABLE', _update_rate=0.5):
        super().__init__()
        self.device = _device
        self.model_type = _model_type
        self.update_rate = _update_rate

        # * determine number of perceived channels
        perception_channels = perception[self.model_type](torch.zeros([1, _channels, 8, 8]).to(_device)).shape[1]
        
        # * determine hidden channels (equalize the parameter count btwn model types)
        hidden_channels = 8*1024 // (perception_channels+_channels)
        hidden_channels = (_hidden+31) // 32*32
        
        # * model layers
        self.conv1 = torch.nn.Conv2d(perception_channels, hidden_channels, 1)
        self.conv2 = torch.nn.Conv2d(hidden_channels, _channels, 1, bias=False)
        with torch.no_grad():
            self.conv2.weight.data.zero_()
        
        # * send to device
        self.to(_device)
        
        # * print model parameter count
        param_n = sum(p.numel() for p in VoxelNCA().parameters())
        print('VoxelNCA param count:', param_n)
        
    def get_alive_mask(_x):
        return func.max_pool2d(_x[:, 3:4, :, :], kernel_size=3, stride=1, padding=1) > 0.1
        
    def forward(self, _x):
        # * get alive mask
        alive_mask = self.get_alive_mask(_x).to(self.device)
        
        # * perception step
        _x = _x.to(self.device)
        p = perception[self.model_type](_x)
        
        # * update step
        p = self.conv2(torch.relu(self.conv1(p)))
        
        # * create stochastic update mask
        stochastic_mask = (torch.rand(_x[:, :1, :, :].shape) <= self.update_rate).to(self.device, torch.float32)
        
        # * perform update
        _x = _x + p * stochastic_mask
        if self.model_type == 'STEERABLE':
            states = _x[:, :-1]*alive_mask
            angle = _x[:, -1:] % (PI*2.0)
            _x = torch.cat([states, angle], 1)
        else:
            _x = _x * alive_mask
        return _x

