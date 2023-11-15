import sys
import os
import json
import torch
import threading

# * add path to '_5_voxel_nca' to sys
cwd = os.getcwd().split('\\')[:-1]
cwd = '/'.join(cwd)
sys.path.insert(1, cwd+'/_5_voxel_nca')

from scripts.nca.VoxelNCA import VoxelNCA as NCA
from scripts.nca import VoxelUtil as util

class NCASimulator:
    def __init__(self, _model, _device='cuda'):
        self.is_running = False
        self.device = _device
        
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
        self.size = _SIZE_+(2*_PAD_)
        self.seed = util.custom_seed(_size=self.size, _channels=params['_CHANNELS_'], _dist=_SEED_DIST_, _center=_SEED_DIC_['center'], 
                                    _plus_x=_SEED_DIC_['plus_x'], _minus_x=_SEED_DIC_['minus_x'],
                                    _plus_y=_SEED_DIC_['plus_y'], _minus_y=_SEED_DIC_['minus_y'],
                                    _plus_z=_SEED_DIC_['plus_z'], _minus_z=_SEED_DIC_['minus_z']).unsqueeze(0).to(self.device)
        self.x = self.seed
        print (f'finished loading model: {_model}')
        
    def run_thread(self):
        while self.is_running:
            with torch.no_grad():
                self.mutex.acquire()
                self.x = self.model(self.x)
                self.mutex.release()
            #print (f'forward {self.count}!')
            #self.count += 1
                
    def run(self):
        # * can only start running if not running
        if self.is_running:
            return
        self.is_running = True
        self.count = 0
        self.mutex = threading.Lock()
        threading.Thread(target=self.run_thread, daemon=True).start()
  
    def stop(self):
        # * can only stop running if running
        if self.is_running:
            self.is_running = False
        
    def get_data(self):
        self.mutex.acquire()
        x = self.x.copy_()
        self.mutex.release()
        