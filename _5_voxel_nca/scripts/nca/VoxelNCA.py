import torch
import torch.nn.functional as func
import datetime
from numpy import pi
from scripts.Video import VideoWriter, zoom
from scripts.vox.Vox import Vox
from scripts.nca.VoxelPerception import VoxelPerception as vp
from scripts.nca import VoxelUtil as util
    
class VoxelNCA(torch.nn.Module):
    def __init__(self, _name, _log_file, _channels=16, _hidden=128, _device='cuda', _model_type='ANISOTROPIC', _update_rate=0.5):
        super().__init__()
        self.device = _device
        self.model_type = _model_type
        self.update_rate = _update_rate
        self.p = vp(_device)
        self.name = _name
        self.log_file = _log_file

        # * determine number of perceived channels
        perception_channels = self.p.perception[self.model_type](self.p, torch.zeros([1, _channels, 8, 8, 8])).shape[1]
        util.logprint(f'_models/{_name}/{_log_file}', f'nca perception channels: {perception_channels}')
        
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
        util.logprint(f'_models/{_name}/{_log_file}', f'nca parameter count: {param_n}')
        
    def generate_video(self, _filename, _seed, _size, _delta=4, _zoom=1, _show_grid=False, _print=True):
        assert _filename != None
        assert _seed != None
        start = datetime.datetime.now()
        with VideoWriter(filename=_filename) as vid:
            # * randomize last channel(s)
            if self.model_type == 'YAW_ISO':
                _seed[:1, -1:] = torch.rand(_size, _size, _size)*pi*2.0
            elif self.model_type == 'QUATERNION':
                pass
                # _seed[:1, -1] = torch.rand(_size, _size, _size)*pi*2.0
                # _seed[:1, -2] = torch.rand(_size, _size, _size)*pi*2.0
                # _seed[:1, -3] = torch.rand(_size, _size, _size)*pi*2.0
                
            x = _seed
            v = Vox().load_from_tensor(x)
            img = v.render(_yaw=_delta, _show_grid=_show_grid, _print=False)
            vid.add(zoom(img, _zoom))
            for i in range(0, 360, _delta):
                for _ in range(_delta):
                    x = self.forward(x)
                v = Vox().load_from_tensor(x)
                img = v.render(_yaw=i, _show_grid=_show_grid, _print=False)
                vid.add(zoom(img, _zoom))
        # * calculate elapsed time
        secs = (datetime.datetime.now()-start).seconds
        elapsed_time = str(datetime.timedelta(seconds=secs))
        util.logprint(f'_models/{self.name}/{self.log_file}', f'created video: {_filename}, gen-time: {elapsed_time}')
    
    def regen_video(self, _filename, _seed, _size, _mask_types=['x+'], _delta=4, _zoom=1, _show_grid=False, _print=True):
        assert _filename != None
        assert _seed != None
        assert _size != None
        start = datetime.datetime.now()
        with VideoWriter(filename=_filename) as vid:
            # * randomize last channel(s)
            if self.model_type == 'YAW_ISO':
                _seed[:1, -1:] = torch.rand(_size, _size, _size)*pi*2.0
            elif self.model_type == 'QUATERNION':
                pass
                # _seed[:1, -1] = torch.rand(_size, _size, _size)*pi*2.0
                # _seed[:1, -2] = torch.rand(_size, _size, _size)*pi*2.0
                # _seed[:1, -3] = torch.rand(_size, _size, _size)*pi*2.0
            
            x = _seed
            v = Vox().load_from_tensor(x)
            img = v.render(_yaw=0, _show_grid=_show_grid, _print=False)
            vid.add(zoom(img, _zoom))
            for i in range(0, 285, _delta):
                for _ in range(_delta):
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
                mask = torch.tensor(util.half_volume_mask(_size, _mask_types[m]))
                x *= mask 
                # * randomize last channel(s)
                if self.model_type == 'YAW_ISO':
                    inv_mask = ~mask
                    x[:1, -1:] += torch.rand(_size, _size, _size)*pi*2.0*inv_mask
                elif self.model_type == 'QUATERNION':
                    pass
                    # inv_mask = ~mask
                    # x[:1, -1] += torch.rand(_size, _size, _size)*pi*2.0*inv_mask
                    # x[:1, -2] += torch.rand(_size, _size, _size)*pi*2.0*inv_mask
                    # x[:1, -3] += torch.rand(_size, _size, _size)*pi*2.0*inv_mask
                
                v = Vox().load_from_tensor(x)
                # * still frames
                img = v.render(_yaw=285, _show_grid=_show_grid, _print=False)
                for i in range(20):
                    vid.add(zoom(img, _zoom))
                # * 360 orbit of regen
                for i in range(0, 360, _delta):
                    for _ in range(_delta):
                        x = self.forward(x)
                    v = Vox().load_from_tensor(x)
                    img = v.render(_yaw=i+285, _show_grid=_show_grid, _print=False)
                    vid.add(zoom(img, _zoom))
        # * calculate elapsed time
        secs = (datetime.datetime.now()-start).seconds
        elapsed_time = str(datetime.timedelta(seconds=secs))
        util.logprint(f'_models/{self.name}/{self.log_file}', f'created video: {_filename}, gen-time: {elapsed_time}')
            
    def rotate_yawiso_video(self, _filename, _seed, _size, _delta=4, _zoom=1, _show_grid=False, _print=True):
        assert _filename != None
        assert _seed != None
        assert _size != None
        start = datetime.datetime.now()
        
        s0 = util.custom_seed(_size=_size, _plus_x='red', _plus_y='green', _plus_z='blue').unsqueeze(0)
        s1 = util.custom_seed(_size=_size, _minus_x='green', _plus_y='red', _plus_z='blue').unsqueeze(0)
        s2 = util.custom_seed(_size=_size, _minus_x='red', _minus_y='green', _plus_z='blue').unsqueeze(0)
        s3 = util.custom_seed(_size=_size, _plus_x='green', _minus_y='red', _plus_z='blue').unsqueeze(0)
        seeds = [s0, s1, s2, s3]
        
        with VideoWriter(filename=_filename) as vid:
            for i in range(len(seeds)):
                x = seeds[i]
                # * randomize last channel(s)
                if self.model_type == 'YAW_ISO':
                    x[:1, -1:] = torch.rand(_size, _size, _size)*pi*2.0
                elif self.model_type == 'QUATERNION':
                    pass
                    # _seed[:1, -1] = torch.rand(_size, _size, _size)*pi*2.0
                    # _seed[:1, -2] = torch.rand(_size, _size, _size)*pi*2.0
                    # _seed[:1, -3] = torch.rand(_size, _size, _size)*pi*2.0

                # * still frames of seed
                v = Vox().load_from_tensor(x)
                img = v.render(_show_grid=_show_grid, _print=False)
                for i in range(32):
                    vid.add(zoom(img, _zoom))
                # * render growth
                for i in range(0, 360, _delta):
                    for _ in range(_delta):
                        x = self.forward(x)
                    v = Vox().load_from_tensor(x)
                    img = v.render(_yaw=i+285, _show_grid=_show_grid, _print=False)
                    vid.add(zoom(img, _zoom))
                img = v.render(_show_grid=_show_grid, _print=False)
                for i in range(32):
                    vid.add(zoom(img, _zoom))
            
            # * calculate elapsed time
            secs = (datetime.datetime.now()-start).seconds
            elapsed_time = str(datetime.timedelta(seconds=secs))
            util.logprint(f'_models/{self.name}/{self.log_file}', f'created video: {_filename}, gen-time: {elapsed_time}')
            
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
        if self.model_type == 'YAW_ISO':
            states = _x[:, :-1]*alive_mask
            angle = _x[:, -1:]%(pi*2.0)
            _x = torch.cat([states, angle], 1)
        elif self.model_type == 'QUATERNION':
            states = _x[:, :-3]*alive_mask
            ax = _x[:, -1:]%(pi*2.0)
            ay = _x[:, -2:]%(pi*2.0)
            az = _x[:, -3:]%(pi*2.0)
            _x = torch.cat([states, az, ay, ax], 1)
        else:
            _x = _x * alive_mask
        if _print: print ('final _x.shape:',_x.shape)
        if _print: print ('********')
        return _x