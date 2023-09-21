import numpy as np
import torch
import torch.nn as nn

class NCA_grow_laplace(nn.Module):
    def __init__(self, _n_channels=16, _hid_channels=128, _fire_rate=0.5, _device=None):
        super().__init__()
        
        self.n_channels = _n_channels
        self.hid_channels = _hid_channels
        self.fire_rate = _fire_rate
        self.device = _device or torch.device('cpu')
        
        # update step
        self.update_module = nn.Sequential(
            nn.Conv2d(
                4*_n_channels,
                _hid_channels,
                kernel_size=1
            ),
            nn.ReLU(),
            nn.Conv2d(
                _hid_channels,
                _n_channels,
                kernel_size=1,
                bias=False
            )
        )
        
        with torch.no_grad():
            self.update_module[2].weight.zero_()
            
        self.to(self.device)
        
    def percieve(self, x, angle=0.0):
        # identity vector
        identity_filter = torch.tensor([[0, 0, 0], [0, 1, 0], [0, 0, 0]], dtype=torch.float32)
        laplace_filter = torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=torch.float32)
        
        # create sobel filters
        scalar = 8.0
        dx = np.outer([1, 2, 1], [-1, 0, 1]) / scalar # Sobel filter
        dy = dx.T
        c, s = np.cos(angle), np.sin(angle)
        sobel_filter_x = torch.tensor(c*dx-s*dy, dtype=torch.float32)
        sobel_filter_y = torch.tensor(s*dx+c*dy, dtype=torch.float32)
        
        # stack filters together
        filters = torch.stack([identity_filter, sobel_filter_x, sobel_filter_y, laplace_filter]) # (filter_num, 3, 3)
        filters = filters.repeat((self.n_channels, 1, 1)) # (filter_num * n_channels, 3, 3)
        self.filters = filters[:, None, ...].to(self.device) # (filter_num * n_channels, 1, 3, 3)
        res = nn.functional.conv2d(x, self.filters, padding=1, groups=self.n_channels)
        return res
        
    def update(self, x):
        return self.update_module(x)
    
    def stochastic_update(self, x, fire_rate):
        device = x.device
        mask = (torch.rand(x[:, :1, :, :].shape) <= fire_rate).to(device, torch.float32)
        return x * mask
    
    def get_living_mask(self, x):
        return (
            nn.functional.max_pool2d(
                x[:, 3:4, :, :], kernel_size=3, stride=1, padding=1
            ) > 0.1
        )
        
    def forward(self, x, angle=0.0):
        pre_life_mask = self.get_living_mask(x)
        
        y = self.percieve(x, angle)
        dx = self.update(y)
        dx = self.stochastic_update(dx, fire_rate=self.fire_rate)
        
        x = x + dx
        
        post_life_mask = self.get_living_mask(x)
        life_mask = (pre_life_mask & post_life_mask).to(torch.float32)
        
        return x * life_mask
    
class NCA_grow_sobel(nn.Module):
    def __init__(self, _n_channels=16, _hid_channels=128, _fire_rate=0.5, _device=None):
        super().__init__()
        
        self.n_channels = _n_channels
        self.hid_channels = _hid_channels
        self.fire_rate = _fire_rate
        self.device = _device or torch.device('cpu')
        
        # update step
        self.update_module = nn.Sequential(
            nn.Conv2d(
                3*_n_channels,
                _hid_channels,
                kernel_size=1
            ),
            nn.ReLU(),
            nn.Conv2d(
                _hid_channels,
                _n_channels,
                kernel_size=1,
                bias=False
            )
        )
        
        with torch.no_grad():
            self.update_module[2].weight.zero_()
            
        self.to(self.device)
        
    def percieve(self, x, angle=0.0):
        # identity vector
        identity_filter = torch.tensor([[0, 0, 0], [0, 1, 0], [0, 0, 0]], dtype=torch.float32)
        
        # create sobel filters
        scalar = 8.0
        dx = np.outer([1, 2, 1], [-1, 0, 1]) / scalar # Sobel filter
        dy = dx.T
        c, s = np.cos(angle), np.sin(angle)
        sobel_filter_x = torch.tensor(c*dx-s*dy, dtype=torch.float32)
        sobel_filter_y = torch.tensor(s*dx+c*dy, dtype=torch.float32)
        
        # stack filters together
        filters = torch.stack([identity_filter, sobel_filter_x, sobel_filter_y]) # (filter_num, 3, 3)
        filters = filters.repeat((self.n_channels, 1, 1)) # (filter_num * n_channels, 3, 3)
        self.filters = filters[:, None, ...].to(self.device) # (filter_num * n_channels, 1, 3, 3)
        res = nn.functional.conv2d(x, self.filters, padding=1, groups=self.n_channels)
        return res
        
    def update(self, x):
        return self.update_module(x)
    
    def stochastic_update(self, x, fire_rate):
        device = x.device
        mask = (torch.rand(x[:, :1, :, :].shape) <= fire_rate).to(device, torch.float32)
        return x * mask
    
    def get_living_mask(self, x):
        return (
            nn.functional.max_pool2d(
                x[:, 3:4, :, :], kernel_size=3, stride=1, padding=1
            ) > 0.1
        )
        
    def forward(self, x, angle=0.0):
        pre_life_mask = self.get_living_mask(x)
        
        y = self.percieve(x, angle)
        dx = self.update(y)
        dx = self.stochastic_update(dx, fire_rate=self.fire_rate)
        
        x = x + dx
        
        post_life_mask = self.get_living_mask(x)
        life_mask = (pre_life_mask & post_life_mask).to(torch.float32)
        
        return x * life_mask