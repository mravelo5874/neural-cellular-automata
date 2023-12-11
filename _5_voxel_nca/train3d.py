import os
import json
import pathlib
import datetime
import numpy as np
import torch
import torch.nn.functional as func
import matplotlib.pylab as pl

from scripts.nca.VoxelNCA import VoxelNCA as NCA
from scripts.nca.VoxelPerception import Perception
from scripts.nca import VoxelUtil as voxutil
from scripts.vox.Vox import Vox

# * target/seed parameters
_NAME_ = 'cowboy16_iso2_init'
_SIZE_ = 16
_PAD_ = 4
_SEED_DIST_ = 4
_SEED_DIC_ = {
    'center': None,
    'plus_x': 'red',
    'minus_x': None,
    'plus_y': 'green',
    'minus_y': None,
    'plus_z': 'blue',
    'minus_z': None,
}
_TARGET_VOX_ = '../vox/cowboy16.vox'
# * model parameters
_MODEL_TYPE_ = Perception.YAW_ISO_V2
_CHANNELS_ = 16
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
_SAVE_RATE_ = 1000

# * load from checkpoint
load_checkpoint = False
checkpoint_dir = '_checkpoints/earth_aniso/'
checkpoint_model = 'earth_aniso_cp10000'
        
def main():
    _FIND_ANOMALY_ = False
    if _FIND_ANOMALY_:
        torch.autograd.set_detect_anomaly(_FIND_ANOMALY_)
        print ('[WARNING] detect anomaly is on. training will be slower than normal.')
    
    
    # * make directory for model files
    if not os.path.exists(f'_models/{_NAME_}'):
        os.mkdir(f'_models/{_NAME_}')

    # * begin logging and start program timer
    voxutil.logprint(f'_models/{_NAME_}/{_LOG_FILE_}', '****************')
    voxutil.logprint(f'_models/{_NAME_}/{_LOG_FILE_}', f'timestamp: {datetime.datetime.now()}')
    voxutil.logprint(f'_models/{_NAME_}/{_LOG_FILE_}', 'initializing training...')
    start = datetime.datetime.now()
    
    # * save model method
    def save_model(_dir, _model, _name):
        # * create directory
        model_path = pathlib.Path(f'{_dir}/{_NAME_}')
        model_path.mkdir(parents=True, exist_ok=True)
        torch.save(_model.state_dict(), f'{model_path.absolute()}/{_name}.pt')
        
        # * save model parameters
        dict = {
            # * target/seed parameters
            '_NAME_': _NAME_,
            '_SIZE_': _SIZE_,
            '_PAD_': _PAD_,
            '_SEED_DIST_': _SEED_DIST_,
            '_SEED_DIC_' : _SEED_DIC_,
            '_TARGET_VOX_': _TARGET_VOX_,
            # * model parameters
            '_MODEL_TYPE_': _MODEL_TYPE_,
            '_CHANNELS_': _CHANNELS_,
            # * training parameters
            '_EPOCHS_': _EPOCHS_,
            '_BATCH_SIZE_': _BATCH_SIZE_,
            '_POOL_SIZE_': _POOL_SIZE_,
            '_UPPER_LR_': _UPPER_LR_,
            '_LOWER_LR_': _LOWER_LR_,
            '_LR_STEP_': _LR_STEP_,
            '_NUM_DAMG_': _NUM_DAMG_,
            '_DAMG_RATE_': _DAMG_RATE_,
            # * logging parameters
            '_INFO_RATE_': _INFO_RATE_,
            '_SAVE_RATE_': _SAVE_RATE_,
        }
        json_object = json.dumps(dict, indent=4)
        with open(f'{model_path.absolute()}/{_name}_params.json', 'w') as outfile:
            outfile.write(json_object)
        voxutil.logprint(f'_models/{_NAME_}/{_LOG_FILE_}', f'model [{_name}] saved to {_dir}...')
    
    # * load model method
    def load_model(_dir, _name):
        # * read params from json file
        with open(f'{_dir}/{_name}/{_name}_params.json', 'r') as openfile:
            global _NAME_
            global _SIZE_
            global _PAD_
            global _SEED_DIST_
            global _SEED_DIC_
            global _TARGET_VOX_
            global _MODEL_TYPE_
            global _CHANNELS_
            global _EPOCHS_
            global _BATCH_SIZE_
            global _POOL_SIZE_
            global _UPPER_LR_
            global _LOWER_LR_
            global _LR_STEP_
            global _NUM_DAMG_
            global _DAMG_RATE_
            global _INFO_RATE_
            global _SAVE_RATE_
            params = json.load(openfile)
            # * target/seed parameters
            _NAME_ = params['_NAME_']+'_from_checkpoint'
            _SIZE_ = params['_SIZE_']
            _PAD_ = params['_PAD_']
            _SEED_DIC_ = params['_SEED_DIC_']
            _SEED_DIST_ = params['_SEED_DIST_']
            _TARGET_VOX_ = params['_TARGET_VOX_']
            # * model parameters
            _MODEL_TYPE_ = params['_MODEL_TYPE_']
            _CHANNELS_ = params['_CHANNELS_']
            # * training parameters
            _EPOCHS_ = params['_EPOCHS_']
            _BATCH_SIZE_ = params['_BATCH_SIZE_']
            _POOL_SIZE_ = params['_POOL_SIZE_']
            _UPPER_LR_ = params['_UPPER_LR_']
            _LOWER_LR_ = params['_LOWER_LR_']
            _LR_STEP_ = params['_LR_STEP_']
            _NUM_DAMG_ = params['_NUM_DAMG_']
            _DAMG_RATE_ = params['_DAMG_RATE_']
            # * logging parameters
            _INFO_RATE_ = params['_INFO_RATE_']
            _SAVE_RATE_ = params['_SAVE_RATE_']
    
    # * sets the device  
    _DEVICE_ = 'cuda' if torch.cuda.is_available() else 'cpu'
    voxutil.logprint(f'_models/{_NAME_}/{_LOG_FILE_}', f'device: {_DEVICE_}')
    torch.backends.cudnn.benchmark = True
    torch.cuda.empty_cache()
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    
    # * create / load model
    if not load_checkpoint:
        model = NCA(_name=_NAME_, _log_file=_LOG_FILE_, _channels=_CHANNELS_, _device=_DEVICE_, _model_type=_MODEL_TYPE_)
        voxutil.logprint(f'_models/{_NAME_}/{_LOG_FILE_}', 'training new model from scratch...')
    else:
        load_model(checkpoint_dir, checkpoint_model)
        model = NCA(_name=_NAME_, _log_file=_LOG_FILE_, _channels=_CHANNELS_, _device=_DEVICE_, _model_type=_MODEL_TYPE_)
        model.load_state_dict(torch.load(checkpoint_dir+'/'+checkpoint_model+'.pt', map_location=_DEVICE_))
        model.train()
        voxutil.logprint(f'_models/{_NAME_}/{_LOG_FILE_}', f'loading checkpoint: {checkpoint_dir}/{checkpoint_model}...')
    
    # * create optimizer and learning-rate scheduler
    opt = torch.optim.Adam(model.parameters(), _UPPER_LR_)
    lr_sched = torch.optim.lr_scheduler.CyclicLR(opt, _LOWER_LR_, _UPPER_LR_, step_size_up=_LR_STEP_, mode='triangular2', cycle_momentum=False)
        
    # * print out parameters
    voxutil.logprint(f'_models/{_NAME_}/{_LOG_FILE_}', f'model: {_NAME_}')
    voxutil.logprint(f'_models/{_NAME_}/{_LOG_FILE_}', f'type: {_MODEL_TYPE_}')
    voxutil.logprint(f'_models/{_NAME_}/{_LOG_FILE_}', f'batch-size: {_BATCH_SIZE_}')
    voxutil.logprint(f'_models/{_NAME_}/{_LOG_FILE_}', f'pool-size: {_POOL_SIZE_}')
    voxutil.logprint(f'_models/{_NAME_}/{_LOG_FILE_}', f'lr: {_UPPER_LR_}>{_LOWER_LR_} w/ {_LR_STEP_} step')

    # * create seed
    PAD_SIZE = _SIZE_+(2*_PAD_)
    seed_ten = voxutil.custom_seed(_size=PAD_SIZE, _channels=_CHANNELS_, _dist=_SEED_DIST_, _center=_SEED_DIC_['center'], 
                                _plus_x=_SEED_DIC_['plus_x'], _minus_x=_SEED_DIC_['minus_x'],
                                _plus_y=_SEED_DIC_['plus_y'], _minus_y=_SEED_DIC_['minus_y'],
                                _plus_z=_SEED_DIC_['plus_z'], _minus_z=_SEED_DIC_['minus_z']).unsqueeze(0).to(_DEVICE_)
    voxutil.logprint(f'_models/{_NAME_}/{_LOG_FILE_}', f'seed.shape: {list(seed_ten.shape)}')
    
    # * load target vox
    target = Vox().load_from_file(_TARGET_VOX_)
    target_ten = target.tensor()
    target_ten = func.pad(target_ten, (_PAD_, _PAD_, _PAD_, _PAD_, _PAD_, _PAD_), 'constant')
    target_ten = target_ten.clone().repeat(_BATCH_SIZE_, 1, 1, 1, 1).to(_DEVICE_)
    voxutil.logprint(f'_models/{_NAME_}/{_LOG_FILE_}', f'target.shape: {list(target_ten.shape)}')
    
    # * create pool
    with torch.no_grad():
        pool = seed_ten.clone().repeat(_POOL_SIZE_, 1, 1, 1, 1)
        # * randomize channel(s)
        if model.isotropic_type() == 1:
            for j in range(_POOL_SIZE_):
                pool[j, -1:] = torch.rand(PAD_SIZE, PAD_SIZE, PAD_SIZE)*np.pi*2.0
        elif model.isotropic_type() == 3:
            for j in range(_POOL_SIZE_):
                pool[j, -1:] = torch.rand(PAD_SIZE, PAD_SIZE, PAD_SIZE)*np.pi*2.0
                pool[j, -2:-1] = torch.rand(PAD_SIZE, PAD_SIZE, PAD_SIZE)*np.pi*2.0
                pool[j, -3:-2] = torch.rand(PAD_SIZE, PAD_SIZE, PAD_SIZE)*np.pi*2.0
    
    # * model training
    voxutil.logprint(f'_models/{_NAME_}/{_LOG_FILE_}', f'starting training w/ {_EPOCHS_+1} epochs...')
    train_start = datetime.datetime.now()
    loss_log = []
    prev_lr = -np.inf
    for i in range(_EPOCHS_+1):
        with torch.no_grad():
            # * sample batch from pool
            batch_idxs = np.random.choice(_POOL_SIZE_, _BATCH_SIZE_, replace=False)
            x = pool[batch_idxs]
            
            # * re-order batch based on loss
            loss_ranks = torch.argsort(voxutil.voxel_wise_loss_function(x, target_ten, _dims=[-1, -2, -3, -4]), descending=True)
            x = x[loss_ranks]
            
            # * re-add seed into batch
            x[:1] = seed_ten
            # * randomize last channel
            if model.isotropic_type() == 1:
                x[:1, -1:] = torch.rand(PAD_SIZE, PAD_SIZE, PAD_SIZE)*np.pi*2.0
            elif model.isotropic_type() == 3:
                x[:1, -1:] = torch.rand(PAD_SIZE, PAD_SIZE, PAD_SIZE)*np.pi*2.0
                x[:1, -2:-1] = torch.rand(PAD_SIZE, PAD_SIZE, PAD_SIZE)*np.pi*2.0
                x[:1, -3:-2] = torch.rand(PAD_SIZE, PAD_SIZE, PAD_SIZE)*np.pi*2.0
        
            # * damage lowest loss in batch
            if i % _DAMG_RATE_ == 0:
                mask = torch.tensor(voxutil.half_volume_mask(PAD_SIZE, 'rand')).to(_DEVICE_)
                # * apply mask
                x[-_NUM_DAMG_:] *= mask
                # * randomize angles for steerable models
                if model.isotropic_type() == 1:
                    inv_mask = ~mask
                    x[-_NUM_DAMG_:, -1:] += torch.rand(PAD_SIZE, PAD_SIZE, PAD_SIZE)*np.pi*2.0*inv_mask
                elif model.isotropic_type() == 3:
                    inv_mask = ~mask
                    x[-_NUM_DAMG_:, -1:] += torch.rand(PAD_SIZE, PAD_SIZE, PAD_SIZE)*np.pi*2.0*inv_mask
                    x[-_NUM_DAMG_:, -2:-1] += torch.rand(PAD_SIZE, PAD_SIZE, PAD_SIZE)*np.pi*2.0*inv_mask
                    x[-_NUM_DAMG_:, -3:-2] += torch.rand(PAD_SIZE, PAD_SIZE, PAD_SIZE)*np.pi*2.0*inv_mask

        # * different loss values
        overflow_loss = 0.0
        diff_loss = 0.0
        target_loss = 0.0
        
        # * forward pass
        num_steps = np.random.randint(64, 96)
        for _ in range(num_steps):
            prev_x = x
            x = model(x)
            diff_loss += (x - prev_x).abs().mean()
            if model.isotropic_type() == 1:
                overflow_loss += (x - x.clamp(-2.0, 2.0))[:, :_CHANNELS_-1].square().sum()
            elif model.isotropic_type() == 3:
                overflow_loss += (x - x.clamp(-2.0, 2.0))[:, :_CHANNELS_-3].square().sum()
            else:
                overflow_loss += (x - x.clamp(-2.0, 2.0))[:, :_CHANNELS_].square().sum()
        
        # * calculate losses
        target_loss += voxutil.voxel_wise_loss_function(x, target_ten)
        target_loss /= 2.0
        diff_loss *= 10.0
        loss = target_loss + overflow_loss + diff_loss
        
        # * backward pass
        with torch.no_grad():
            loss.backward()
            # * normalize gradients 
            for p in model.parameters():
                p.grad /= (p.grad.norm()+1e-5)

            torch.nn.utils.clip_grad_norm_(model.parameters(), 5) # maybe? : 
            opt.step()
            opt.zero_grad()
            lr_sched.step()
            # * re-add batch to pool
            pool[batch_idxs] = x
            # * correctly add to loss log
            _loss = loss.item()
            if torch.isnan(loss) or torch.isinf(loss) or torch.isneginf(loss):
                pass
            else:
                loss_log.append(_loss)
                
            # * detect invalid loss values :(
            if torch.isnan(loss) or torch.isinf(loss) or torch.isneginf(loss):
                voxutil.logprint(f'_models/{_NAME_}/{_LOG_FILE_}', f'detected invalid loss value: {loss}')
                voxutil.logprint(f'_models/{_NAME_}/{_LOG_FILE_}', f'overflow loss: {overflow_loss}, diff loss: {diff_loss}, target loss: {target_loss}')
                raise ValueError
            
            # * print info
            if i % _INFO_RATE_ == 0 and i!= 0:
                secs = (datetime.datetime.now()-train_start).seconds
                time = str(datetime.timedelta(seconds=secs))
                iter_per_sec = float(i)/float(secs)
                est_time_sec = int((_EPOCHS_-i)*(1/iter_per_sec))
                est = str(datetime.timedelta(seconds=est_time_sec))
                avg = sum(loss_log[-_INFO_RATE_:])/float(_INFO_RATE_)
                lr = np.round(lr_sched.get_last_lr()[0], 8)
                step = '▲'
                if prev_lr > lr:
                    step = '▼'
                prev_lr = lr
                voxutil.logprint(f'_models/{_NAME_}/{_LOG_FILE_}', f'[{i}/{_EPOCHS_+1}]\t {np.round(iter_per_sec, 3)}it/s\t time: {time}~{est}\t loss: {np.round(avg, 3)}>{np.round(np.min(loss_log), 3)}\t lr: {lr} {step}')
            
            # * save checkpoint
            if i % _SAVE_RATE_ == 0 and i != 0:
                save_model('_checkpoints', model, _NAME_+'_cp'+str(i))
                
    # * print train time
    secs = (datetime.datetime.now()-train_start).seconds
    train_time = str(datetime.timedelta(seconds=secs))
    voxutil.logprint(f'_models/{_NAME_}/{_LOG_FILE_}', f'train time: {train_time}')
    
    # * save loss plot
    pl.plot(loss_log, '.', alpha=0.1)
    pl.yscale('log')
    pl.ylim(np.min(loss_log), loss_log[0])
    pl.savefig(f'_models/{_NAME_}/{_NAME_}_loss_plot.png')
                
    # * save final model
    save_model('_models', model, _NAME_)
    
    # * generate videos
    s = _SIZE_+(2*_PAD_)
    curr = datetime.datetime.now().strftime("%H:%M:%S")
    # voxutil.logprint(f'_models/{_NAME_}/{_LOG_FILE_}', f'starting time: {curr}')
    # voxutil.logprint(f'_models/{_NAME_}/{_LOG_FILE_}', 'generating videos...')
    # with torch.no_grad():
    #     model.generate_video(f'_models/{_NAME_}/vidtrain_grow.mp4', seed_ten, _size=s)
    #     if _NUM_DAMG_ > 0:
    #         model.regen_video(f'_models/{_NAME_}/vidtrain_multi_regen.mp4', seed_ten, _size=s, _mask_types=['x+', 'y+', 'z+'])
    #     if model.isotropic_type() > 0:
    #         model.rotate_yawiso_video(f'_models/{_NAME_}/vidtrain_multi_rotate.mp4', seed_ten, _size=s, _show_grid=True)
    
    # * calculate elapsed time
    secs = (datetime.datetime.now()-start).seconds
    elapsed_time = str(datetime.timedelta(seconds=secs))
    voxutil.logprint(f'_models/{_NAME_}/{_LOG_FILE_}', f'elapsed time: {elapsed_time}')
    voxutil.logprint(f'_models/{_NAME_}/{_LOG_FILE_}', '****************')
    
if __name__ == '__main__':
    main()