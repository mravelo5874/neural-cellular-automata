import os
import torch
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
# * custom imports
from nca_model import nca_model
from nca_trainer import nca_trainer
from nca_perception import ptype
from nca_util import *

nca_params = {
    # * target/seed parameters
    '_NAME_': 'minicube5_test',
    '_NOTE_': '''Testing new training script (_9_post_thesis_optimizations)''',
    '_SIZE_': 5,
    '_PAD_': 3,
    '_USE_SPHERE_SEED_': False,
    '_SEED_POINTS_': 2,
    '_SEED_DIST_': 1,
    '_SEED_DIC_': {
        'center': None,
        'plus_x': 'red',
        'minus_x': None,
        'plus_y': 'green',
        'minus_y': None,
        'plus_z': 'blue',
        'minus_z': None
    },
    '_SEED_HID_INFO_': False,
    '_TARGET_VOX_': '../vox/minicube5.vox',
    # * model parameters
    '_PTYPE_': ptype.ANISOTROPIC,
    '_CHANNELS_': 16,
    '_HIDDEN_': 128,
    # * training parameters
    '_EPOCHS_': 5_000,
    '_BATCH_SIZE_': 4,
    '_POOL_SIZE_': 32,
    '_UPPER_LR_': 1e-3,
    '_LOWER_LR_': 1e-5,
    '_LR_STEP_': 2000,
    '_NUM_DAMG_': 2,
    '_DAMG_RATE_': 5,
    # * logging parameters
    '_LOG_FILE_': 'trainlog.txt',
    '_INFO_RATE_': 100,
    '_SAVE_RATE_': 5000,
}

def ddp_setup(_rank: int, _world_size: int):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '11235'
    init_process_group(backend='nccl', rank=_rank, world_size=_world_size)

def run(_rank: int, _world_size: int):
    ddp_setup(_rank, _world_size)
    
    # * regularly used params
    name = nca_params['_NAME_']
    logf = nca_params['_LOG_FILE_']
    
    # * make directory for model files
    if not os.path.exists(f'models/{name}'):
        os.mkdir(f'models/{name}')
    
    # * print cuda devices
    devices = []
    logprint(f'models/{name}/{logf}', '========================')
    logprint(f'models/{name}/{logf}', 'available cuda devices:')
    for i in range (torch.cuda.device_count()):
        devices.append(i)
        prop = torch.cuda.get_device_properties(i)
        mem = prop.total_memory // (1024**2)
        mpcount = prop.multi_processor_count
        logprint(f'models/{name}/{logf}', f'{i}: {torch.cuda.get_device_name(i)}, mem:{mem}MB, mpc:{mpcount}')
    logprint(f'models/{name}/{logf}', '========================')
    
    # * create model, optimizer, and lr-scheduler
    vanilla_model = nca_model(_channels=nca_params['_CHANNELS_'], _hidden=nca_params['_HIDDEN_'], _ptype=nca_params['_PTYPE_'])
    ddp_model = DDP(vanilla_model, device_ids=[])
    optim = torch.optim.Adam(vanilla_model.parameters(), nca_params['_UPPER_LR_'])
    sched = torch.optim.lr_scheduler.CyclicLR(optim, nca_params['_LOWER_LR_'], nca_params['_UPPER_LR_'], step_size_up=nca_params['_LR_STEP_'], mode='triangular2', cycle_momentum=False)
    
    # * get seed and target tensors
    seed_ten = generate_seed(nca_params)
    target_ten = load_vox_as_tensor(nca_params)
    logprint(f'models/{name}/{logf}', f'seed.shape: {list(seed_ten.shape)}')
    logprint(f'models/{name}/{logf}', f'target.shape: {list(target_ten.shape)}')
    
    # * create pool tensor
    isotype = vanilla_model.pobj.orientation_channels(vanilla_model.ptype)
    pool = generate_pool(nca_params, seed_ten, isotype)
    
    # * create trainer and begin training 
    trainer = nca_trainer(ddp_model, optim, sched, seed_ten, target_ten, pool, nca_params, isotype)
    trainer.begin()
    destroy_process_group()

def main():
    # * setup distributed data parallel and spawn workers
    world_size = torch.cuda.device_count()
    mp.spawn(run, args=(world_size), nprocs=world_size)

if __name__ == '__main__':
    main()