import numpy as np
import torch
import torch.nn as nn

"""
LEARNABLE MODEL - uses a learnable 2d convolution within preception
"""
class NCA_grow_learnable(nn.Module):
    def __init__(self, _n_channels=16, _hid_channels=128, _fire_rate=0.5, _device=None, zero_bias=True):
        super().__init__()
        
        self.n_channels = _n_channels
        self.hid_channels = _hid_channels
        self.fire_rate = _fire_rate
        self.device = _device or torch.device('cpu')
        
        # perception step
        self.percieve_module = nn.Conv2d(
            _n_channels,
            _n_channels*3,
            3,
            stride=1,
            padding=1,
            groups=_n_channels,
            bias=False
        )
        
        def init_weights(m):
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.02)
                if getattr(m, "bias", None) is not None:
                    if zero_bias: nn.init.zeros_(m.bias)
                    else: nn.init.normal_(m.bias, std=0.02)
        
        # with torch.no_grad():
        #     self.apply(init_weights)
        
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
        return self.percieve_module(x)
        
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
    
"""
MAGNITUDE MODEL - uses laplace, sobel_mag, and identity within preception
"""
class NCA_grow_magnitude(nn.Module):
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
        scalar = 8.0
        dx = np.outer([1, 2, 1], [-1, 0, 1]) / scalar # sobel filter
        dy = dx.T
        c, s = np.cos(angle), np.sin(angle)
        sobel_x = torch.tensor(c*dx-s*dy, dtype=torch.float32)
        sobel_y = torch.tensor(s*dx+c*dy, dtype=torch.float32)
        laplace = torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=torch.float32)
        laplace /= scalar
        identty = torch.tensor([[0, 0, 0], [0, 1, 0], [0, 0, 0]], dtype=torch.float32)

        sobel_x = sobel_x.repeat((self.n_channels, 1, 1)) # (n_channels, 3, 3)
        sobel_y = sobel_y.repeat((self.n_channels, 1, 1)) # (n_channels, 3, 3)
        laplace = laplace.repeat((self.n_channels, 1, 1)) # (n_channels, 3, 3)
        identty = identty.repeat((self.n_channels, 1, 1)) # (n_channels, 3, 3)

        sobel_x = sobel_x[:, None, ...].to(self.device) # (n_channels, 1, 3, 3)
        sobel_y = sobel_y[:, None, ...].to(self.device) # (n_channels, 1, 3, 3)
        laplace = laplace[:, None, ...].to(self.device) # (n_channels, 1, 3, 3)
        identty = identty[:, None, ...].to(self.device) # (n_channels, 1, 3, 3)

        # 2d convs
        G_x = nn.functional.conv2d(x, sobel_x, padding=1, groups=self.n_channels)
        G_y = nn.functional.conv2d(x, sobel_y, padding=1, groups=self.n_channels)
        G_l = nn.functional.conv2d(x, laplace, padding=1, groups=self.n_channels)
        G_i = nn.functional.conv2d(x, identty, padding=1, groups=self.n_channels)

        for i in range(self.n_channels):
            min_y = torch.min(G_y[:, i])
            max_y = torch.max(G_y[:, i])
            # Shift the values from [-min, max] to [0, abs(min_y)+abs(max_y)]
            G_y[:, i] = G_y[:, i]+abs(min_y)
            # Scale the values to [0, 1]
            div_y = (abs(min_y)+abs(max_y))
            if div_y != 0:
                G_y[:, i] = G_y[:, i]/div_y
            
            min_x = torch.min(G_x[:, i])
            max_x = torch.max(G_x[:, i])
            # Shift the values from [-min, max] to [0, abs(min_y)+abs(max_y)]
            G_x[:, i] = G_x[:, i]+abs(min_x)
            # Scale the values to [0, 1]
            div_x = (abs(min_x)+abs(max_x))
            if div_x != 0:
                G_x[:, i] = G_x[:, i]/div_x

        G_x2 = torch.square(G_x)
        G_y2 = torch.square(G_y)
        G_mag = torch.zeros_like(x)
        for i in range(0, self.n_channels):
            G_mag[:,i] = torch.sqrt(G_x2[:,i]+G_y2[:,i])
        res = torch.cat([G_i, G_l, G_mag], dim=1)
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
        
        with torch.no_grad():
            y = self.percieve(x, angle)
        dx = self.update(y)
        dx = self.stochastic_update(dx, fire_rate=self.fire_rate)
        
        x = x + dx
        
        post_life_mask = self.get_living_mask(x)
        life_mask = (pre_life_mask & post_life_mask).to(torch.float32)
        
        return x * life_mask

"""
LAPLACE MODEL - uses laplace, sobel_x, sobel_y, and identity within preception
"""
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

"""
ORIGINAL MODEL - uses sobel_x, sobel_y, and identity within preception
"""
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