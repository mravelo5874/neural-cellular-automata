import torch
import torch.nn.functional as func
from numpy import pi
from _9_post_thesis_optimizations.nca_perception import ptype, nca_perception

class nca_model(torch.nn.Module):
    def __init__(
        self, 
        _channels=16, 
        _hidden=128, 
        _rate=0.5, 
        _ptype=ptype.ANISOTROPIC):
        super().__init__()
        
        # * save variables
        self.rate = _rate
        self.ptype = _ptype
        self.pobj = nca_perception()
        self.pfunc = self.pobj.get_function[_ptype]
        
        # * calculate hidden channel values
        perception_channels = self.pfunc(self.pobj, torch.zeros([1, _channels, 8, 8, 8])).shape[1]
        hidden_channels = 8*1024 // (perception_channels+_channels)
        hidden_channels = (_hidden+31) // 32*32
        
        # * model layers
        self.conv1 = torch.nn.Conv3d(perception_channels, hidden_channels, 1)
        self.relu = torch.nn.ReLU(inplace=True)
        self.conv2 = torch.nn.Conv3d(hidden_channels, _channels, 1, bias=False)
        with torch.no_grad():
            self.conv2.weight.data.zero_()
            
    def get_alive_mask(self, _x):
        return func.max_pool3d(_x[:, 3:4, :, :, :], kernel_size=3, stride=1, padding=1) > 0.1
    
    def forward(self, _x):
        # * send to device
        x = _x.to(self.device)
        
        # * get alive mask
        alive_mask = self.get_alive_mask(x).to(self.device)
    
        # * perception step
        p = self.pfunc(self.p, x)
        
        # * update step
        p = self.conv1(p)
        p = self.relu(p)
        p = self.conv2(p)
        
        # * create stochastic mask
        stochastic_mask = (torch.rand(_x[:, :1, :, :, :].shape) <= self.rate).to(self.device, torch.float32)
        
        # * perform stochastic update
        x = x + p * stochastic_mask
        
        # * final isotropic concatination + apply alive mask
        ori_channels = self.pobj.orientation_channels(self.ptype)
        if ori_channels == 1:
            states = x[:, :-1]*alive_mask
            angle = x[:, -1:] % (pi*2.0)
            x = torch.cat([states, angle], 1)
            
        elif ori_channels == 3:
            states = x[:, :-3]*alive_mask
            ax = x[:, -1:] % (pi*2.0)
            ay = x[:, -2:-1] % (pi*2.0)
            az = x[:, -3:-2] % (pi*2.0)
            x = torch.cat([states, az, ay, ax], 1)
            
        else:
            x = x * alive_mask
           
        return x