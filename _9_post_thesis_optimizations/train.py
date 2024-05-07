import os
import numpy as np

import torch
import torch.nn.functional as func
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

from nca_model import nca_model
from nca_trainer import nca_trainer
from nca_perception import nca_perception, ptype

from scripts.nca import VoxelUtil as voxutil
from scripts.vox.Vox import Vox

# * target/seed parameters
_NAME_ = 'sphere16_isoRmat_thesis'
_NOTE_ = '''...'''
_SIZE_ = 16
_PAD_ = 4
_USE_SPHERE_SEED_ = False
_SEED_POINTS_ = 2
_SEED_DIST_ = 2
_SEED_DIC_ = {
    'center': None,
    'plus_x': 'red',
    'minus_x': None,
    'plus_y': 'green',
    'minus_y': None,
    'plus_z': 'blue',
    'minus_z': None
}
_SEED_HID_INFO_ = False
_TARGET_VOX_ = '../voxnp/sphere16.npy'
# * model parameters
_PTYPE_ = ptype.ANISOTROPIC
_CHANNELS_ = 16
_HIDDEN_ = 128
# * training parameters
_EPOCHS_ = 10_000
_BATCH_SIZE_ = 4
_POOL_SIZE_ = 32
_UPPER_LR_ = 5e-4
_LOWER_LR_ = 1e-5
_LR_STEP_ = 2000
_NUM_DAMG_ = 2
_DAMG_RATE_ = 5
# * logging parameters
_LOG_FILE_ = 'trainlog.txt'
_INFO_RATE_ = 100
_SAVE_RATE_ = 5000

def ddp_setup(_rank, _world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '11235'
    init_process_group(backend='nccl', rank=_rank, world_size=_world_size)

def main():
    
    # * create model, optimizer, and lr-scheduler
    vanilla_model = nca_model(_channels=_CHANNELS_, _hidden=_HIDDEN_, _ptype=_PTYPE_)
    ddp_model = DDP(vanilla_model, device_ids=[])
    optim = torch.optim.Adam(vanilla_model.parameters(), _UPPER_LR_)
    sched = torch.optim.lr_scheduler.CyclicLR(optim, _LOWER_LR_, _UPPER_LR_, step_size_up=_LR_STEP_, mode='triangular2', cycle_momentum=False)
    
    # * create seed tensor
    PAD_SIZE = _SIZE_+(2*_PAD_)
    if _USE_SPHERE_SEED_:
        seed_ten = voxutil.seed_3d(_size=PAD_SIZE, _channels=_CHANNELS_, _points=_SEED_POINTS_, _radius=_SEED_DIST_).unsqueeze(0)
    else:
        seed_ten = voxutil.custom_seed(_size=PAD_SIZE, _channels=_CHANNELS_, _dist=_SEED_DIST_, _hidden_info=_SEED_HID_INFO_,
                                    _center=_SEED_DIC_['center'], 
                                    _plus_x=_SEED_DIC_['plus_x'], _minus_x=_SEED_DIC_['minus_x'],
                                    _plus_y=_SEED_DIC_['plus_y'], _minus_y=_SEED_DIC_['minus_y'],
                                    _plus_z=_SEED_DIC_['plus_z'], _minus_z=_SEED_DIC_['minus_z']).unsqueeze(0)
    voxutil.logprint(f'_models/{_NAME_}/{_LOG_FILE_}', f'seed.shape: {list(seed_ten.shape)}')
    
    # * load target vox and create target tensor
    if _TARGET_VOX_.endswith('vox'):
        target = Vox().load_from_file(_TARGET_VOX_)
        target_ten = target.tensor()
    elif _TARGET_VOX_.endswith('npy'):
        with open(_TARGET_VOX_, 'rb') as f:
            target_ten = torch.from_numpy(np.load(f))
    
    target_ten = func.pad(target_ten, (_PAD_, _PAD_, _PAD_, _PAD_, _PAD_, _PAD_), 'constant')
    target_ten = target_ten.clone().repeat(_BATCH_SIZE_, 1, 1, 1, 1)
    voxutil.logprint(f'_models/{_NAME_}/{_LOG_FILE_}', f'target.shape: {list(target_ten.shape)}')
    
    # * create pool tensor
    ISO_TYPE = nca_perception.orientation_channels(nca_perception(), _PTYPE_)
    with torch.no_grad():
        pool = seed_ten.clone().repeat(_POOL_SIZE_, 1, 1, 1, 1)
        # * randomize channel(s)
        if ISO_TYPE == 1:
            for j in range(_POOL_SIZE_):
                pool[j, -1:] = torch.rand(PAD_SIZE, PAD_SIZE, PAD_SIZE)*np.pi*2.0
        elif ISO_TYPE == 3:
            for j in range(_POOL_SIZE_):
                pool[j, -1:] = torch.rand(PAD_SIZE, PAD_SIZE, PAD_SIZE)*np.pi*2.0
                pool[j, -2:-1] = torch.rand(PAD_SIZE, PAD_SIZE, PAD_SIZE)*np.pi*2.0
                pool[j, -3:-2] = torch.rand(PAD_SIZE, PAD_SIZE, PAD_SIZE)*np.pi*2.0
    
    # * create trainer and begin training 
    trainer = nca_trainer(ddp_model, optim, sched, seed_ten, target_ten, pool)
    trainer.begin()

if __name__ == '__main__':
    main()