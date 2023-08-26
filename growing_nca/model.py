import torch
import torch.nn as nn

class NCA_model(nn.Module):
    """
    Parameters
    ----------
    n_channels : int
        Number of channels in the grid.
    
    hid_channels : int
        Number of hidden channels that are related to the pixel-wise 1x1 convolution.
        
    fire_rate : float
        Number between 0 and 1. The lower it is the more likely it is for cells to be 
        set to 0 during the 'stochastic_update' process.
        
    device : torch.device
        What device we perform all the computations.
        
    Attributes
    ----------
    update_module : nn.Sequential
        Single part of the network containing trainable parameters. Composed of a 1x1
        convolution, ReLU, and another 1x1 convolution.
    
    filters : torch.Tensor
        Constant tensor of shape (3 * n_channels, 1, 3, 3)
    """
    
    def __init__(self, _n_channels=16, _hid_channels=128, _fire_rate=0.5, _device=None):
        super().__init__()
        
        self.n_channels = _n_channels
        self.hid_channels = _hid_channels
        self.fire_rate = _fire_rate
        self.device = _device or torch.device('cpu')
        
        # perceieve step - manually create filters
        sobel_filter = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
        scalar = 8.0

        sobel_filter_x = sobel_filter / scalar
        sobel_filter_y = sobel_filter.t() / scalar
        identity_filter = torch.tensor([[0, 0, 0], [0, 1, 0], [0, 0, 0]], dtype=torch.float)
        filters = torch.stack([identity_filter, sobel_filter_x, sobel_filter_y]) # (3, 3, 3)
        filters = filters.repeat((_n_channels, 1, 1)) # (3 * n_channels, 3, 3)
        self.filters = filters[:, None, ...].to(self.device) # (3 * n_channels, 1, 3, 3)
        
        # update step
        self.update_module = nn.Sequential(
            nn.Conv2d(
                3 * _n_channels,
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
        
    def percieve(self, x):
        """
        Approximates channel-wise gradient and combines it with the input.
        
        This is the only place where we include information on the neighboring cells.
        However, we are not using any learnable parameters here.
        
        Parameters
        ----------
        x : torch.Tensor
            Shape `(n_samples, n_channels, grid_size, grid_size)`.
            
        Returns
        -------
        torch.Tensor
            Shape `(n_samples, 3 * n_channels, grid_size, grid_size)`.
        """
        return nn.functional.conv2d(x, self.filters, padding=1, groups=self.n_channels)
    
    def update(self, x):
        """
        Performs an update.
        
        Note that this is the only part of the forward pass that uses trainable parameters.
       
        Parameters
        ----------
        x : torch.Tensor
            Shape `(n_samples, 3 * n_channels, grid_size, grid_size)`.
            
        Returns
        -------
        torch.Tensor
            Shape `(n_samples, n_channels, grid_size, grid_size)`.
        """
        return self.update_module(x)
    
    #@staticmethod
    def stochastic_update(self, x, fire_rate):
        """
        Runs pixel-wise dropout.
        
        Unlike dropout, there is no scaling taking place.
        
        Parameters
        ----------
        x : torch.Tensor
            Shape `(n_samples, n_channels, grid_size, grid_size)`.
            
        fire_rate : float
            Number between 0 and 1. The lower it is the more likely a given cell updates.
        
        Returns
        -------
        torch.Tensor
            Shape `(n_samples, n_channels, grid_size, grid_size)`.
        """
        device = x.device
        mask = (torch.rand(x[:, :1, :, :].shape) <= fire_rate).to(device, torch.float32)
        return x * mask
    
    #@staticmethod
    def get_living_mask(self, x):
        """
        Identifies living cells.
        
        Parameters
        ----------
        x : torch.Tensor
            Shape `(n_samples, n_channels, grid_size, grid_size)`.
            
        Returns
        -------
        torch.Tensor
            Shape `(n_samples, 1, grid_size, grid_size)` and the dtype is bool.
        """
        return (
            nn.functional.max_pool2d(
                x[:, 3:4, :, :], kernel_size=3, stride=1, padding=1
            ) > 0.1
        )
        
    def forward(self, x):
        """
        Run the forward pass. One iteration of the rule.
        
        Parameters
        ----------
        x : torch.Tensor
            Shape `(n_samples, n_channels, grid_size, grid_size)`.
            
        Returns
        -------
        torch.Tensor
            Shape `(n_samples, n_channels, grid_size, grid_size)`.
        """
        pre_life_mask = self.get_living_mask(x)
        
        y = self.percieve(x)
        dx = self.update(y)
        dx = self.stochastic_update(dx, fire_rate=self.fire_rate)
        
        x = x + dx
        
        post_life_mask = self.get_living_mask(x)
        life_mask = (pre_life_mask & post_life_mask).to(torch.float32)
        
        return x * life_mask