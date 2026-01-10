import os
import datetime
import numpy as np
import torch
import torch.nn.functional as func
import matplotlib.pylab as pl

from scripts.nca.VoxelNCA import VoxelNCA as NCA
from scripts.nca.VoxelPerception import Perception
from scripts.nca import VoxelUtil as voxutil
from scripts.vox.Vox import Vox

# * stat paramaters
_TRIALS_ = 3
_SUCC_LOSS_THRESH_ = 1.0

# * target/seed parameters
_NAME_ = 'minicube5_v0'
_SIZE_ = 5
_PAD_ = 3
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
_TARGET_VOX_ = '../vox/minicube5.vox'
# * model parameters
_MODEL_TYPE_ = Perception.YAW_ISO_V3
_CHANNELS_ = 16
_HIDDEN_ = 128
# * training parameters
_EPOCHS_ = 3_000
_BATCH_SIZE_ = 4
_POOL_SIZE_ = 32
_UPPER_LR_ = 1e-3
_LOWER_LR_ = 1e-5
_LR_STEP_ = 1000
_NUM_DAMG_ = 2
_DAMG_RATE_ = 5
# * logging parameters
_LOG_FILE_ = f'{_NAME_}_trials_log.txt'
_INFO_RATE_ = 100

def force_cudnn_initialization():
    voxutil.logprint(f'_trials/{_LOG_FILE_}', 'forcing cuDNN initialization...')
    s = 32
    dev = torch.device('cuda')
    torch.nn.functional.conv2d(torch.zeros(s, s, s, s, device=dev), torch.zeros(s, s, s, s, device=dev))
        
def main():

    # * begin logging and start program timer
    voxutil.logprint(f'_trials/{_LOG_FILE_}', '****************')
    voxutil.logprint(f'_trials/{_LOG_FILE_}', f'timestamp: {datetime.datetime.now()}')
    voxutil.logprint(f'_trials/{_LOG_FILE_}', 'initializing trials...')
    start = datetime.datetime.now()
    
    # * sets the device  
    _DEVICE_ = 'cuda' if torch.cuda.is_available() else 'cpu'
    voxutil.logprint(f'_trials/{_LOG_FILE_}', f'device: {_DEVICE_}')
    torch.backends.cudnn.benchmark = True
    torch.cuda.empty_cache()
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    force_cudnn_initialization()

    # * begin running trials
    min_losses = []
    for T in range(_TRIALS_):

        voxutil.logprint(f'_trials/{_LOG_FILE_}', '------------------------')
        voxutil.logprint(f'_trials/{_LOG_FILE_}', f'beginning trial {T}...')
        trial_start = datetime.datetime.now()
    
        # * create / load model
        model = NCA(_name=_NAME_, _log_file=None, _channels=_CHANNELS_, _hidden=_HIDDEN_, _device=_DEVICE_, _model_type=_MODEL_TYPE_)
        voxutil.logprint(f'_trials/{_LOG_FILE_}', 'training new model from scratch...')

        # * save model isotropic type
        ISO_TYPE = model.isotropic_type()    
        
        # * create optimizer and learning-rate scheduler
        opt = torch.optim.Adam(model.parameters(), _UPPER_LR_)
        lr_sched = torch.optim.lr_scheduler.CyclicLR(opt, _LOWER_LR_, _UPPER_LR_, step_size_up=_LR_STEP_, mode='triangular2', cycle_momentum=False)
            
        # * print out parameters
        voxutil.logprint(f'_trials/{_LOG_FILE_}', f'model: {_NAME_}')
        voxutil.logprint(f'_trials/{_LOG_FILE_}', f'type: {_MODEL_TYPE_}')
        voxutil.logprint(f'_trials/{_LOG_FILE_}', f'channels: {_CHANNELS_}')
        voxutil.logprint(f'_trials/{_LOG_FILE_}', f'hidden: {_HIDDEN_}')
        voxutil.logprint(f'_trials/{_LOG_FILE_}', f'hidden-seed-info: {_SEED_HID_INFO_}')
        voxutil.logprint(f'_trials/{_LOG_FILE_}', f'batch-size: {_BATCH_SIZE_}')
        voxutil.logprint(f'_trials/{_LOG_FILE_}', f'pool-size: {_POOL_SIZE_}')
        voxutil.logprint(f'_trials/{_LOG_FILE_}', f'lr: {_UPPER_LR_}>{_LOWER_LR_} w/ {_LR_STEP_} step')

        # * create seed
        PAD_SIZE = _SIZE_+(2*_PAD_)
        if _USE_SPHERE_SEED_:
            seed_ten = voxutil.seed_3d(_size=PAD_SIZE, _channels=_CHANNELS_, _points=_SEED_POINTS_, _radius=_SEED_DIST_).unsqueeze(0).to(_DEVICE_)
        else:
            seed_ten = voxutil.custom_seed(_size=PAD_SIZE, _channels=_CHANNELS_, _dist=_SEED_DIST_, _hidden_info=_SEED_HID_INFO_,
                                        _center=_SEED_DIC_['center'], 
                                        _plus_x=_SEED_DIC_['plus_x'], _minus_x=_SEED_DIC_['minus_x'],
                                        _plus_y=_SEED_DIC_['plus_y'], _minus_y=_SEED_DIC_['minus_y'],
                                        _plus_z=_SEED_DIC_['plus_z'], _minus_z=_SEED_DIC_['minus_z']).unsqueeze(0).to(_DEVICE_)
        voxutil.logprint(f'_trials/{_LOG_FILE_}', f'seed.shape: {list(seed_ten.shape)}')
        
        # * load target vox
        if _TARGET_VOX_.endswith('vox'):
            target = Vox().load_from_file(_TARGET_VOX_)
            target_ten = target.tensor()
        elif _TARGET_VOX_.endswith('npy'):
            with open(_TARGET_VOX_, 'rb') as f:
                target_ten = torch.from_numpy(np.load(f))
        
        target_ten = func.pad(target_ten, (_PAD_, _PAD_, _PAD_, _PAD_, _PAD_, _PAD_), 'constant')
        target_ten = target_ten.clone().repeat(_BATCH_SIZE_, 1, 1, 1, 1).to(_DEVICE_)
        voxutil.logprint(f'_trials/{_LOG_FILE_}', f'target.shape: {list(target_ten.shape)}')
        
        # * create pool
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
        
        # * model training
        voxutil.logprint(f'_trials/{_LOG_FILE_}', f'starting training w/ {_EPOCHS_+1} epochs...')
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
                if ISO_TYPE == 1:
                    x[:1, -1:] = torch.rand(PAD_SIZE, PAD_SIZE, PAD_SIZE)*np.pi*2.0
                elif ISO_TYPE == 3:
                    x[:1, -1:] = torch.rand(PAD_SIZE, PAD_SIZE, PAD_SIZE)*np.pi*2.0
                    x[:1, -2:-1] = torch.rand(PAD_SIZE, PAD_SIZE, PAD_SIZE)*np.pi*2.0
                    x[:1, -3:-2] = torch.rand(PAD_SIZE, PAD_SIZE, PAD_SIZE)*np.pi*2.0
            
                # * damage lowest loss in batch
                if i % _DAMG_RATE_ == 0:
                    mask = torch.tensor(voxutil.half_volume_mask(PAD_SIZE, 'rand')).to(_DEVICE_)
                    # * apply mask
                    x[-_NUM_DAMG_:] *= mask
                    # * randomize angles for steerable models
                    if ISO_TYPE == 1:
                        inv_mask = ~mask
                        x[-_NUM_DAMG_:, -1:] += torch.rand(PAD_SIZE, PAD_SIZE, PAD_SIZE)*np.pi*2.0*inv_mask
                    elif ISO_TYPE == 3:
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
                if ISO_TYPE == 1:
                    overflow_loss += (x - x.clamp(-2.0, 2.0))[:, :_CHANNELS_-1].square().sum()
                elif ISO_TYPE == 3:
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
                    voxutil.logprint(f'_trials/{_LOG_FILE_}', f'detected invalid loss value: {loss}')
                    voxutil.logprint(f'_trials/{_LOG_FILE_}', f'overflow loss: {overflow_loss}, diff loss: {diff_loss}, target loss: {target_loss}')
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
                    voxutil.logprint(f'_trials/{_LOG_FILE_}', f'[{i}/{_EPOCHS_+1}]\t {np.round(iter_per_sec, 3)}it/s\t time: {time}~{est}\t loss: {np.round(avg, 3)}>{np.round(np.min(loss_log), 3)}\t lr: {lr} {step}')
                    
        # * print train time
        secs = (datetime.datetime.now()-train_start).seconds
        train_time = str(datetime.timedelta(seconds=secs))
        voxutil.logprint(f'_trials/{_LOG_FILE_}', f'train time: {train_time}')

        min_losses.append(np.round(np.min(loss_log), 3))
        
        # * save loss plot
        pl.plot(loss_log, '.', alpha=0.1)
        pl.yscale('log')
        pl.ylim(np.min(loss_log), loss_log[0])
        pl.savefig(f'_trials/{_LOG_FILE_}_trial_{T}_loss_plot.png')
    
        # * calculate trial elapsed time
        secs = (datetime.datetime.now()-trial_start).seconds
        elapsed_time = str(datetime.timedelta(seconds=secs))
        voxutil.logprint(f'_trials/{_LOG_FILE_}', f'elapsed time for trial {T}: {elapsed_time}')
    
    # * calculate total elapsed time
    secs = (datetime.datetime.now()-start).seconds
    elapsed_time = str(datetime.timedelta(seconds=secs))
    voxutil.logprint(f'_trials/{_LOG_FILE_}', f'elapsed time: {elapsed_time}')
    voxutil.logprint(f'_trials/{_LOG_FILE_}', '****************')

    count = 0
    for i in min_losses:
        if i < _SUCC_LOSS_THRESH_:
            count += 1

    voxutil.logprint(f'_trials/{_LOG_FILE_}', f'trials success rate for thresh of {_SUCC_LOSS_THRESH_}: {np.round(count/len(min_losses), 3)}')
    
if __name__ == '__main__':
    main()