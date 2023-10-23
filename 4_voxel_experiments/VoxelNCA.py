import torch
import torch.nn.functional as func
from Vox import Vox
from VoxelPerception import VoxelPerception as vp
from numpy import pi as PI
from Video import VideoWriter, zoom

def voxel_wise_loss_function(_x, _target, _scale=1e3, _dims=[]):
    return _scale * torch.mean(torch.square(_x[:, :4] - _target), _dims)

def create_seed(_size=16, _channels=16, _dist=5, _points=4):
    x = torch.zeros([_channels, _size, _size, _size])
    half = _size//2
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
    
class VoxelNCA(torch.nn.Module):
    def __init__(self, _channels=16, _hidden=128, _device='cuda', _model_type='ANISOTROPIC', _update_rate=0.5):
        super().__init__()
        self.device = _device
        self.model_type = _model_type
        self.update_rate = _update_rate
        self.p = vp(_device)

        # * determine number of perceived channels
        perception_channels = self.p.perception[self.model_type](self.p, torch.zeros([1, _channels, 8, 8, 8])).shape[1]
        print ('perception_channels:',perception_channels)
        
        # * determine hidden channels (equalize the parameter count btwn model types)
        hidden_channels = 8*1024 // (perception_channels+_channels)
        hidden_channels = (_hidden+31) // 32*32
        
        # * model layers
        self.conv1 = torch.nn.Conv3d(perception_channels, hidden_channels, 1)
        self.conv2 = torch.nn.Conv3d(hidden_channels, _channels, 1, bias=False)
        with torch.no_grad():
            self.conv2.weight.data.zero_()
        
        # * send to device
        self.to(_device)
        
        # * print model parameter count
        param_n = sum(p.numel() for p in self.parameters())
        print('VoxelNCA param count:', param_n)
        
    def generate_video(self, _filename, _seed, _steps=128, _delta=0, _zoom=1, _show_grid=False, _print=True):
        assert _filename != None
        assert _seed != None
        with VideoWriter(filename=_filename) as vid:
            x = _seed
            for i in range(_steps):
                x = self.forward(x)
                v = Vox().load_from_tensor(x)
                img = v.render(_yaw=_delta*i, _show_grid=_show_grid, _print=_print)
                vid.add(zoom(img, _zoom))
            if _print: vid.show()
        
    def get_alive_mask(self, _x):
        return func.max_pool3d(_x[:, 3:4, :, :, :], kernel_size=3, stride=1, padding=1) > 0.1
        
    def forward(self, _x, _print=False):
        if _print: print ('init _x.shape:',_x.shape)
        # * get alive mask
        alive_mask = self.get_alive_mask(_x).to(self.device)
        if _print: print ('init alive_mask.shape:',alive_mask.shape)
        
        # * perception step
        _x = _x.to(self.device)
        p = self.p.perception[self.model_type](self.p, _x)
        if _print: print ('perception p.shape:',p.shape)
        
        # * update step
        p = self.conv2(torch.relu(self.conv1(p)))
        if _print: print ('update p.shape:',p.shape)
        
        # * create stochastic update mask
        stochastic_mask = (torch.rand(_x[:, :1, :, :, :].shape) <= self.update_rate).to(self.device, torch.float32)
        if _print: print ('stochastic_mask.shape:',stochastic_mask.shape)
        
        # * perform update
        _x = _x + p * stochastic_mask
        if self.model_type == 'STEERABLE':
            states = _x[:, :-1]*alive_mask
            angle = _x[:, -1:] % (PI*2.0)
            _x = torch.cat([states, angle], 1)
        else:
            _x = _x * alive_mask
        if _print: print ('final _x.shape:',_x.shape)
        if _print: print ('********')
        return _x

