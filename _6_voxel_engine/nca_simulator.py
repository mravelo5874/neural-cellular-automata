import sys
import os
import json
import time
import torch
import threading
import numpy as np

# * add path to '_5_voxel_nca' to sys
cwd = os.getcwd().split('\\')[:-1]
cwd = '/'.join(cwd)
sys.path.insert(1, cwd+'/_5_voxel_nca')

from scripts.nca.VoxelNCA import VoxelNCA as NCA
from scripts.nca import VoxelUtil as util

class NCASimulator:
    def __init__(self, _model, _device='cuda'):
        self.is_loaded = False
        self.is_running = False
        self.is_paused = False
        self.started = False
        self.device = _device
        self.mutex = threading.Lock()
        
        # * setup cuda if available
        torch.backends.cudnn.benchmark = True
        if self.device == 'cuda' and torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
        else:
            self.device = 'cpu'
            
        # * load nca model
        self.load_model(_model)
        
    def load_model(self, _model):
        # * load in params for seed
        params = {}
        with open(f'{cwd}/models/{_model}/{_model}_params.json', 'r') as openfile:
            params = json.load(openfile)
        model = NCA(_name=params['_NAME_'], _channels=params['_CHANNELS_'], _device=self.device, _model_type=params['_MODEL_TYPE_'])
        model.load_state_dict(torch.load(f'{cwd}/models/{_model}/{_model}.pt', map_location=self.device))
        model.eval()
        self.model = model
            
        # * create seed
        _SIZE_ = params['_SIZE_']
        _PAD_ = params['_PAD_']
        _SEED_DIST_ = params['_SEED_DIST_']
        _SEED_DIC_ = params['_SEED_DIC_']
        self.size = _SIZE_+(2*_PAD_)*2
        self.seed = util.custom_seed(_size=self.size, _channels=params['_CHANNELS_'], _dist=_SEED_DIST_, _center=_SEED_DIC_['center'], 
                                    _plus_x=_SEED_DIC_['plus_x'], _minus_x=_SEED_DIC_['minus_x'],
                                    _plus_y=_SEED_DIC_['plus_y'], _minus_y=_SEED_DIC_['minus_y'],
                                    _plus_z=_SEED_DIC_['plus_z'], _minus_z=_SEED_DIC_['minus_z']).unsqueeze(0).to(self.device)
        # * randomize channels
        if model.isotropic_type() == 1:
                self.seed[:1, -1:] = torch.rand(self.size, self.size, self.size)*np.pi*2.0
        elif model.isotropic_type() == 3:
            self.seed[:1, -1:] = torch.rand(self.size, self.size, self.size)*np.pi*2.0
            self.seed[:1, -2:-1] = torch.rand(self.size, self.size, self.size)*np.pi*2.0
            self.seed[:1, -3:-2] = torch.rand(self.size, self.size, self.size)*np.pi*2.0
            
        # * set tensor
        self.mutex.acquire()
        self.x = self.seed.detach().clone()
        self.mutex.release()
        self.is_loaded = True
        
    def run_thread(self, _delay):
        # * wait 3 seconds to show off seed
        time.sleep(_delay)
        self.is_running = True
        while self.is_running:
            with torch.no_grad():
                self.mutex.acquire()
                if not self.is_paused:
                    self.x = self.model(self.x)
                    self.started = True
                self.mutex.release()
                
    def run(self, _delay=0.5):
        # * can only start running if not running
        if self.is_running:
            return
        self.worker = threading.Thread(target=self.run_thread, args=[_delay], daemon=True)
        self.worker.start()
        
    def step_forward(self):
        with torch.no_grad():
            self.mutex.acquire()
            if self.is_paused:
                self.x = self.model(self.x)
            self.mutex.release()
            
    def rot_seed(self, _axis='z'):
        # can only rotate seed if not started
        if not self.started:
            if _axis == 'x':
                self.seed = torch.rot90(self.seed, 1, (2, 3))
            elif _axis == 'y':
                self.seed = torch.rot90(self.seed, 1, (2, 4))
            elif _axis == 'z':
                self.seed = torch.rot90(self.seed, 1, (3, 4))
            self.mutex.acquire()
            self.x = self.seed.detach().clone()
            self.mutex.release()
     
    def stop(self):
        self.is_running = False
        self.mutex.acquire()
        self.is_paused = False
        self.started = False
        self.mutex.release()
        self.worker.join()
        
    def unload(self):
        # * can only unload if loaded and running
        if self.is_loaded and self.is_running:
            self.is_loaded = False
            self.stop()
            return True
        return False
  
    def toggle_pause(self):
        # * can only pause if running
        if self.is_running:
            self.mutex.acquire()
            self.is_paused = not self.is_paused
            self.mutex.release()

    def reset(self):
        # * can only reset if running
        if self.is_running:
            self.is_running = False
            if self.worker:
                self.worker.join()
            self.mutex.acquire()
            self.started = False
            self.x = self.seed.detach().clone()
            self.mutex.release()
            self.run()
        
    def get_data(self):
        # * get tensor
        self.mutex.acquire()
        data = self.x.cpu().detach().numpy()
        self.mutex.release()
        
        # convert numbers to -> np.uint8 values between 0-255
        data = data[:, :4, ...]
        data = np.clip(data, 0.0, 1.0)
        data = np.transpose(data, (0, 2, 4, 3, 1))
        _, x, y, z, rgba = data.shape
        data = data.reshape((x*y*z, rgba))*255
        data = data.astype(np.uint8)
        return data.tobytes()
    
    def get_cubes(self):
        # * copy x as numpy array
        self.mutex.acquire()
        vol = np.array(self.x.cpu())
        self.mutex.release()
        
        # * get voxels that are alive
        vol = vol.squeeze(0)[3, ...]
        cubes = np.argwhere(vol > 0.1)
        return cubes, self.size
    
    def erase_sphere(self, _pos, _radius):
        X, Y, Z = np.ogrid[:self.size, :self.size, :self.size]
        dist_from_center = np.sqrt((X-_pos[0])**2 + (Y-_pos[1])**2 + (Z-_pos[2])**2)
        mask = torch.tensor(dist_from_center >= _radius)
        
        self.mutex.acquire()
        self.x = self.x*mask
        self.mutex.release()