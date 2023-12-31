import torch
import torch.nn.functional as func
from Vox import Vox
from VoxelPerception import VoxelPerception as vp
from VoxelUtil import VoxelUtil as util
from numpy import pi as PI
from Video import VideoWriter, zoom
    
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

    def is_steerable(self):
        return self.model_type == 'YAW_ISO'
        
    def generate_video(self, _filename, _seed, _delta=2, _zoom=1, _show_grid=False, _print=True):
        assert _filename != None
        assert _seed != None
        with VideoWriter(filename=_filename) as vid:
            x = _seed
            v = Vox().load_from_tensor(x)
            img = v.render(_yaw=_delta, _show_grid=_show_grid, _print=False)
            vid.add(zoom(img, _zoom))
            for i in range(0, 360, _delta):
                x = self.forward(x)
                v = Vox().load_from_tensor(x)
                img = v.render(_yaw=i, _show_grid=_show_grid, _print=False)
                vid.add(zoom(img, _zoom))
            if _print: vid.show()
    
    def regen_video(self, _filename, _seed, _size, _mask_types=['x+'], _zoom=1, _show_grid=False, _print=True):
        assert _filename != None
        assert _seed != None
        assert _size != None
        with VideoWriter(filename=_filename) as vid:
            x = _seed
            v = Vox().load_from_tensor(x)
            img = v.render(_yaw=0, _show_grid=_show_grid, _print=False)
            vid.add(zoom(img, _zoom))
            for i in range(0, 285, 2):
                x = self.forward(x)
                v = Vox().load_from_tensor(x)
                img = v.render(_yaw=i, _show_grid=_show_grid, _print=False)
                vid.add(zoom(img, _zoom))
            for m in range(len(_mask_types)):
                # * still frames
                img = v.render(_yaw=285, _show_grid=_show_grid, _print=False)
                for i in range(20):
                    vid.add(zoom(img, _zoom))
                # * apply mask
                mask = util.half_volume_mask(_size, _mask_types[m])
                x *= torch.tensor(mask)
                v = Vox().load_from_tensor(x)
                # * still frames
                img = v.render(_yaw=285, _show_grid=_show_grid, _print=False)
                for i in range(20):
                    vid.add(zoom(img, _zoom))
                # * 360 orbit of regen
                for i in range(0, 360, 2):
                    x = self.forward(x)
                    v = Vox().load_from_tensor(x)
                    img = v.render(_yaw=i+285, _show_grid=_show_grid, _print=False)
                    vid.add(zoom(img, _zoom))
            if _print: vid.show()
            
    def rotate_video(self, _filename, _seed, _size, _rot_types=[(4, 3), (2, 3), (2, 3)], _zoom=1, _show_grid=False, _print=True):
        assert _filename != None
        assert _seed != None
        assert _size != None
        with VideoWriter(filename=_filename) as vid:
            # * still frames of seed
            x = _seed
            v = Vox().load_from_tensor(x)
            img = v.render(_show_grid=_show_grid, _print=False)
            for i in range(32):
                vid.add(zoom(img, _zoom))
            # * render growth
            for i in range(0, 360, 2):
                x = self.forward(x)
                v = Vox().load_from_tensor(x)
                img = v.render(_yaw=i+285, _show_grid=_show_grid, _print=False)
                vid.add(zoom(img, _zoom))
            img = v.render(_show_grid=_show_grid, _print=False)
            for i in range(32):
                vid.add(zoom(img, _zoom))
            for r in range(len(_rot_types)):
                # * still frames of seed
                x = torch.rot90(_seed, 1, _rot_types[r])
                v = Vox().load_from_tensor(x)
                img = v.render(_show_grid=_show_grid, _print=False)
                for i in range(32):
                    vid.add(zoom(img, _zoom))
                # * render growth
                for i in range(0, 360, 2):
                    x = self.forward(x)
                    v = Vox().load_from_tensor(x)
                    img = v.render(_yaw=i+285, _show_grid=_show_grid, _print=False)
                    vid.add(zoom(img, _zoom))
                img = v.render(_show_grid=_show_grid, _print=False)
                for i in range(32):
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
        if self.is_steerable():
            states = _x[:, :-1]*alive_mask
            angle = _x[:, -1:] % (PI*2.0)
            _x = torch.cat([states, angle], 1)
        else:
            _x = _x * alive_mask
        if _print: print ('final _x.shape:',_x.shape)
        if _print: print ('********')
        return _x

