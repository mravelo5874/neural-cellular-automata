import os
import json
import pathlib
import datetime
import numpy as np
import torch
import torch.nn.functional as func
import matplotlib.pylab as pl

from scripts.nca.VoxelNCA import VoxelNCA as NCA
from scripts.nca import VoxelUtil as util
from scripts.vox.Vox import Vox

def main():
    print ('****************')
    print ('initializing training...')
    start = datetime.datetime.now()
    
    # * target/seed parameters
    _NAME_ = 'cowboy16_aniso'
    _SIZE_ = 16
    _PAD_ = 4
    _SEED_POINTS_ = 4
    _SEED_DIST_ = 4
    _TARGET_VOX_ = '../_vox/cowboy16.vox'
    # * model parameters
    _MODEL_TYPE_ = 'ANISOTROPIC'
    _CHANNELS_ = 16
    # * training parameters
    _EPOCHS_ = 10_000
    _BATCH_SIZE_ = 4
    _POOL_SIZE_ = 32
    _UPPER_LR_ = 1e-3
    _LOWER_LR_ = 1e-5
    _NUM_DAMG_ = 2
    _DAMG_RATE_ = 5
    # * logging parameters
    _INFO_RATE_ = 250
    _SAVE_RATE_ = 1000
    _VIDEO_RATE_ = 100_000
    
    # * print out parameters
    print (f'model: {_NAME_}')
    print (f'type: {_MODEL_TYPE_}')
    print (f'batch size: {_BATCH_SIZE_}')
    print (f'pool size: {_POOL_SIZE_}')
    print (f'lr: {_UPPER_LR_} > {_LOWER_LR_}')
    
    # * sets the device  
    _DEVICE_ = 'cuda' if torch.cuda.is_available() else 'cpu'
    print ('device:', _DEVICE_)
    torch.backends.cudnn.benchmark = True
    torch.cuda.empty_cache()
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    
    # * save model method
    def save_model(_dir, _model, _name):
        model_path = pathlib.Path(_dir)
        model_path.mkdir(parents=True, exist_ok=True)
        torch.save(_model.state_dict(), _dir + '/' + _name + '.pt')
        
        # * save model parameters
        dict = {
            # * target/seed parameters
            '_NAME_': _NAME_,
            '_SIZE_': _SIZE_,
            '_PAD_': _PAD_,
            '_SEED_POINTS_': _SEED_POINTS_,
            '_SEED_DIST_': _SEED_DIST_,
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
            '_NUM_DAMG_': _NUM_DAMG_,
            '_DAMG_RATE_': _DAMG_RATE_,
            # * logging parameters
            '_INFO_RATE_': _INFO_RATE_,
            '_SAVE_RATE_': _SAVE_RATE_,
            '_VIDEO_RATE_': _VIDEO_RATE_,
        }
        json_object = json.dumps(dict, indent=4)
        with open(_dir + '/' + _name + '_params.json', 'w') as outfile:
            outfile.write(json_object)
        print (f'\'{_name}\' saved to {_dir}...')
    
    # * create model
    model = NCA(_channels=_CHANNELS_, _device=_DEVICE_, _model_type=_MODEL_TYPE_)
    opt = torch.optim.Adam(model.parameters(), _UPPER_LR_)
    lr_sched = torch.optim.lr_scheduler.CyclicLR(opt, _LOWER_LR_, _UPPER_LR_, step_size_up=1000, mode='triangular2', cycle_momentum=False)

    # * create seed
    seed_ten = util.create_seed(_size=_SIZE_+(2*_PAD_), _dist=_SEED_DIST_, _points=_SEED_POINTS_).unsqueeze(0).to(_DEVICE_)
    
    # * load target vox
    target = Vox().load_from_file(_TARGET_VOX_)
    target_ten = target.tensor()
    target_ten = func.pad(target_ten, (_PAD_, _PAD_, _PAD_, _PAD_, _PAD_, _PAD_), 'constant')
    target_ten = target_ten.clone().repeat(_BATCH_SIZE_, 1, 1, 1, 1).to(_DEVICE_)
    
    # * create pool
    with torch.no_grad():
        pool = seed_ten.clone().repeat(_POOL_SIZE_, 1, 1, 1, 1)
    
    # * model training
    print (f'starting training w/ {_EPOCHS_+1} epochs...')
    train_start = datetime.datetime.now()
    loss_log = []
    prev_lr = -np.inf
    for i in range(_EPOCHS_+1):
        with torch.no_grad():
            # * sample batch from pool
            batch_idxs = np.random.choice(_POOL_SIZE_, _BATCH_SIZE_, replace=False)
            x = pool[batch_idxs]
            
            # * re-order batch based on loss
            loss_ranks = torch.argsort(util.voxel_wise_loss_function(x, target_ten, _dims=[-1, -2, -3, -4]), descending=True)
            x = x[loss_ranks]
            
            # * re-add seed into batch
            x[:1] = seed_ten
            
            # * damage lowest loss in batch
            if i % _DAMG_RATE_ == 0:
                mask = util.half_volume_mask(_SIZE_+(2*_PAD_), 'rand')
                x[-_NUM_DAMG_:] *= torch.tensor(mask).to(_DEVICE_)

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
            if model.is_steerable():
                overflow_loss += (x - x.clamp(-2.0, 2.0))[:, :15].square().sum()
            else:
                overflow_loss += (x - x.clamp(-2.0, 2.0))[:, :16].square().sum()
        
        # * calculate losses
        target_loss += util.voxel_wise_loss_function(x, target_ten)
        target_loss /= 2.0
        diff_loss *= 10.0
        loss = target_loss + overflow_loss + diff_loss
        
        # * backward pass
        with torch.no_grad():
            loss.backward()
            # * normalize gradients 
            for p in model.parameters():
                p.grad /= (p.grad.norm()+1e-8)
            #torch.nn.utils.clip_grad_norm(model.parameters(), 5) # maybe? : 
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

            # nan loss :9
            if torch.isnan(loss) or torch.isinf(loss) or torch.isneginf(loss):
                print (f'detected invalid loss value: {loss}')
                raise ValueError
            
            # * print info
            if i % _INFO_RATE_ == 0 and i!= 0:
                secs = (datetime.datetime.now()-train_start).seconds
                time = str(datetime.timedelta(seconds=secs))
                iter_per_sec = float(i)/float(secs)
                est_time_sec = int((_EPOCHS_-i)*(1/iter_per_sec))
                est = str(datetime.timedelta(seconds=est_time_sec))
                lr = np.round(lr_sched.get_last_lr()[0], 10)
                step = '▲'
                if prev_lr > lr:
                    step = '▼'
                prev_lr = lr
                print(f'[iter {i}] iter/sec: {np.round(iter_per_sec, 3)}\t time: {time}\t est: {est}')
                print(f'           loss>min: {np.round(_loss, 3)} > {np.round(np.min(loss_log), 3)}\t lr: {lr} {step}')
                                
            # * save checkpoint
            if i % _SAVE_RATE_ == 0 and i != 0:
                save_model('_checkpoints', model, _NAME_+'_cp'+str(i))
                
            # * create video
            if i % _VIDEO_RATE_ == 0 and i != 0:
                model.generate_video(f'_videos/{_NAME_}_cp{i}.mp4', seed_ten)

    # * print train time
    secs = (datetime.datetime.now()-train_start).seconds
    train_time = str(datetime.timedelta(seconds=secs))
    print (f'train time: {train_time}')
    
    # * save loss plot
    pl.plot(loss_log, '.', alpha=0.1)
    pl.yscale('log')
    pl.ylim(np.min(loss_log), loss_log[0])
    pl.savefig(f'_models/{_NAME_}_loss_plot.png')
                
    # * save final model
    save_model('_models', model, _NAME_)
    
    # * create videos
    print ('generating videos...')
    with torch.no_grad():
        model.generate_video(f'_videos/{_NAME_}_grow.mp4', seed_ten)
        model.regen_video(f'_videos/{_NAME_}_multi_regen.mp4', seed_ten, _size=_SIZE_+(2*_PAD_), _mask_types=['x+', 'y+', 'z+'])
        model.rotate_video(f'_videos/{_NAME_}_multi_rotate.mp4', seed_ten, _size=_SIZE_+(2*_PAD_))
    
    # * calculate elapsed time
    secs = (datetime.datetime.now()-start).seconds
    elapsed_time = str(datetime.timedelta(seconds=secs))
    print (f'elapsed time: {elapsed_time}')
    print ('****************')
    
if __name__ == '__main__':
    main()