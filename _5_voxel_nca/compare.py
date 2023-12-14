import os
import json
import torch
import random
import datetime
import numpy as np

from scripts.nca.VoxelNCA import VoxelNCA as NCA
from scripts.nca import VoxelUtil as voxutil
from scripts.vox.Vox import Vox

# * compair params
_NAME_ = 'ghosty_iso2_v12' # yawiso_10
_DIR_ = '_models'
_DEVICE_ = 'cuda'
_LOG_FILE_ = 'complog.txt'
_ITER_ = 1

# * for reproducability
_SEED_ = 100
np.random.seed(_SEED_)
random.seed(_SEED_)
torch.manual_seed(_SEED_)

def main():
    # * make directory for model files
    if not os.path.exists(f'_models/{_NAME_}'):
        os.mkdir(f'_models/{_NAME_}')
    
    # * begin logging and start program timer
    voxutil.logprint(f'_models/{_NAME_}/{_LOG_FILE_}', '****************')
    voxutil.logprint(f'_models/{_NAME_}/{_LOG_FILE_}', f'timestamp: {datetime.datetime.now()}')
    voxutil.logprint(f'_models/{_NAME_}/{_LOG_FILE_}', 'initializing comparison...')
    start = datetime.datetime.now()
    
    # * set cuda as device
    voxutil.logprint(f'_models/{_NAME_}/{_LOG_FILE_}', f'device: {_DEVICE_}')
    torch.backends.cudnn.benchmark = True
    torch.cuda.empty_cache()
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    
    # * load in params for seed
    params = {}
    with open(f'{_DIR_}/{_NAME_}/{_NAME_}_params.json', 'r') as openfile:
        params = json.load(openfile)
        _SIZE_ = params['_SIZE_']
        _PAD_ = params['_PAD_']
        _SEED_DIST_ = params['_SEED_DIST_']
        _SEED_DIC_ = params['_SEED_DIC_']
        
    # * load model
    path = f'{_DIR_}/{_NAME_}/{_NAME_}.pt'
    model = NCA(_name=_NAME_, _log_file=_LOG_FILE_, _channels=params['_CHANNELS_'], _model_type=params['_MODEL_TYPE_'])
    model.load_state_dict(torch.load(path, map_location=_DEVICE_))   
    model.eval()
    voxutil.logprint(f'_models/{_NAME_}/{_LOG_FILE_}', f'loaded model from: {path}')
        
    # * create seed
    PAD_SIZE = _SIZE_+(2*_PAD_)
    seed = voxutil.custom_seed(_size=PAD_SIZE, _channels=params['_CHANNELS_'], _dist=_SEED_DIST_, _center=_SEED_DIC_['center'], 
                                _plus_x=_SEED_DIC_['plus_x'], _minus_x=_SEED_DIC_['minus_x'],
                                _plus_y=_SEED_DIC_['plus_y'], _minus_y=_SEED_DIC_['minus_y'],
                                _plus_z=_SEED_DIC_['plus_z'], _minus_z=_SEED_DIC_['minus_z']).unsqueeze(0).to(_DEVICE_)
    voxutil.logprint(f'_models/{_NAME_}/{_LOG_FILE_}', f'seed.shape: {seed.shape}')
    voxutil.logprint(f'_models/{_NAME_}/{_LOG_FILE_}', f'model.iso_type: {model.isotropic_type()}')
    
    # * create seed 0
    seed_0 = seed.detach().clone()
    
    # * randomize channel(s)
    if model.isotropic_type() == 1:
        seed_0[:, -1:] = torch.rand(PAD_SIZE, PAD_SIZE, PAD_SIZE)*np.pi*2.0
    elif model.isotropic_type() == 3:
        seed_0[:, -1:] = torch.rand(PAD_SIZE, PAD_SIZE, PAD_SIZE)*np.pi*2.0
        seed_0[:, -2:-1] = torch.rand(PAD_SIZE, PAD_SIZE, PAD_SIZE)*np.pi*2.0
        seed_0[:, -3:-2] = torch.rand(PAD_SIZE, PAD_SIZE, PAD_SIZE)*np.pi*2.0

    # * create seed 1
    seed_1 = seed_0.detach().clone()
    
    # * rotate seed 1!
    seed_1 = torch.rot90(seed_1, 1, (2, 3))
    
    # * rotate istropic channel(s)
    if model.isotropic_type() == 1:
        seed_1[:, -1:] = torch.add(seed_1[:, -1:], (np.pi/2))
    elif model.isotropic_type() == 3:
        seed_1[:, -1:] = torch.add(seed_1[:, -1:], (np.pi/2))
        seed_1[:, -2:-1] = torch.add(seed_1[:, -2:-1],(np.pi/2))
        seed_1[:, -3:-2] = torch.add(seed_1[:, -3:-2],(np.pi/2))

    # * compare seeds
    seed_1_copy = seed_1.detach().clone()

    # * rotate istropic channel(s)
    if model.isotropic_type() == 1:
        seed_1_copy[:, -1:] = torch.sub(seed_1_copy[:, -1:], (np.pi/2))
    elif model.isotropic_type() == 3:
        seed_1_copy[:, -1:] = torch.sub(seed_1_copy[:, -1:], (np.pi/2))
        seed_1_copy[:, -2:-1] = torch.sub(seed_1_copy[:, -2:-1], (np.pi/2))
        seed_1_copy[:, -3:-2] = torch.sub(seed_1_copy[:, -3:-2], (np.pi/2))
        
    seed_1_copy = torch.rot90(seed_1_copy, 1, (3, 2))
    
    # Vox().load_from_tensor(seed_0).render(_show_grid=True)
    # Vox().load_from_tensor(seed_1_copy).render(_show_grid=True)
    
    dif = torch.abs(seed_0 - seed_1_copy)
    res = torch.all(dif < 0.0001)
    print (f'pre-comp: {res}')
    
    # * run forward once
    with torch.no_grad():
        for i in range(_ITER_):
            mask = (torch.rand(seed_0[:, :1, :, :, :].shape) <= model.update_rate).to(model.device, torch.float32)
            seed_0 = model.forward(seed_0, _mask=mask, _comp=seed_1)
            
            # * rotate mask
            rot_mask = torch.rot90(mask, 1, (2, 3))
            seed_1 = model.forward(seed_1, _mask=rot_mask)
            
            # * compare seeds
            seed_1_copy = seed_1.detach().clone()
            seed_1_copy = torch.rot90(seed_1, 1, (3, 2))
    
            # * rotate istropic channel(s)
            if model.isotropic_type() == 1:
                seed_1_copy[:, -1:] = torch.sub(seed_1_copy[:, -1:], (np.pi/2))
            elif model.isotropic_type() == 3:
                seed_1_copy[:, -1:] = torch.sub(seed_1_copy[:, -1:], (np.pi/2))
                seed_1_copy[:, -2:-1] = torch.sub(seed_1_copy[:, -2:-1], (np.pi/2))
                seed_1_copy[:, -3:-2] = torch.sub(seed_1_copy[:, -3:-2], (np.pi/2))
            
            dif = torch.abs(seed_0 - seed_1_copy)
            res = torch.all(dif < 0.0001)
            print (f'{i} comp: {res}')
            
            # dif += seed
            # Vox().load_from_tensor(dif).render(_show_grid=True)
            
    # * compare seeds
    seed_1_copy = seed_1.detach().clone()
    seed_1_copy = torch.rot90(seed_1, 1, (3, 2))
    
    # * rotate istropic channel(s)
    if model.isotropic_type() == 1:
        seed_1_copy[:, -1:] = torch.sub(seed_1_copy[:, -1:], (np.pi/2))
    elif model.isotropic_type() == 3:
        seed_1_copy[:, -1:] = torch.sub(seed_1_copy[:, -1:], (np.pi/2))
        seed_1_copy[:, -2:-1] = torch.sub(seed_1_copy[:, -2:-1], (np.pi/2))
        seed_1_copy[:, -3:-2] = torch.sub(seed_1_copy[:, -3:-2], (np.pi/2))
    
    dif = torch.abs(seed_0 - seed_1_copy)
    res = torch.all(dif < 0.0001)
    print (f'post-comp: {res}')
    
    # * calculate elapsed time
    secs = (datetime.datetime.now()-start).seconds
    elapsed_time = str(datetime.timedelta(seconds=secs))
    voxutil.logprint(f'_models/{_NAME_}/{_LOG_FILE_}', f'elapsed time: {elapsed_time}')
    voxutil.logprint(f'_models/{_NAME_}/{_LOG_FILE_}', '****************')

if __name__ == '__main__':
    main()