import os
import json
import pathlib
import datetime
import numpy as np
import torch
import torch.nn.functional as func

from scripts import Utility as util
from scripts.IsoNCA import IsoNCA as NCA

# * target/seed parameters
_NAME_ = 'cowboy_angle'
# * model parameters
_MODEL_TYPE_ = 'ANGLE_STEER'
_CHANNELS_ = 16
_SIZE_ = 40
_PAD_ = 12
_SEED_POINTS_ = 1
_SEED_DIST_ = 4
_TARGET_IMG_ = '../_images/cowboy.png'
# * training parameters
_EPOCHS_ = 10_000
_BATCH_SIZE_ = 8
_POOL_SIZE_ = 32
_UPPER_LR_ = 1e-3
_LOWER_LR_ = 1e-5
_LR_STEP_ = 2000
_NUM_DAMG_ = 2
_DAMG_RATE_ = 5
# * logging parameters
_LOG_FILE_ = 'trainlog.txt'
_INFO_RATE_ = 200
_SAVE_RATE_ = 1000

# * load from checkpoint
load_checkpoint = False
checkpoint_dir = ''
checkpoint_model = ''

def main():
    # * make directory for model files
    if not os.path.exists(f'_models/{_NAME_}'):
        os.mkdir(f'_models/{_NAME_}')
        
    # * begin logging and start program timer
    util.logprint(f'_models/{_NAME_}/{_LOG_FILE_}', '****************')
    util.logprint(f'_models/{_NAME_}/{_LOG_FILE_}', f'timestamp: {datetime.datetime.now()}')
    util.logprint(f'_models/{_NAME_}/{_LOG_FILE_}', 'initializing training...')
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
        }
        json_object = json.dumps(dict, indent=4)
        with open(f'{model_path.absolute()}/{_name}_params.json', 'w') as outfile:
            outfile.write(json_object)
        util.logprint(f'_models/{_NAME_}/{_LOG_FILE_}', f'model [{_name}] saved to {_dir}...')
        
    # * load model method
    def load_model(_dir, _name):
        # * read params from json file
        with open(f'{_dir}/{_name}/{_name}_params.json', 'r') as openfile:
            global _NAME_
            params = json.load(openfile)
            # * target/seed parameters
            _NAME_ = params['_NAME_']+'_from_checkpoint'
            
    # * create / load model
    if not load_checkpoint:
        model = NCA(_name=_NAME_, _log_file=_LOG_FILE_, _channels=_CHANNELS_, _device=_DEVICE_, _model_type=_MODEL_TYPE_)
        util.logprint(f'_models/{_NAME_}/{_LOG_FILE_}', 'training new model from scratch...')
    else:
        load_model(checkpoint_dir, checkpoint_model)
        model = NCA(_name=_NAME_, _log_file=_LOG_FILE_, _channels=_CHANNELS_, _device=_DEVICE_, _model_type=_MODEL_TYPE_)
        model.load_state_dict(torch.load(checkpoint_dir+'/'+checkpoint_model+'.pt', map_location=_DEVICE_))
        model.train()
        util.logprint(f'_models/{_NAME_}/{_LOG_FILE_}', f'loading checkpoint: {checkpoint_dir}/{checkpoint_model}...')
    
    # * create optimizer and learning-rate scheduler
    opt = torch.optim.Adam(model.parameters(), _UPPER_LR_)
    lr_sched = torch.optim.lr_scheduler.CyclicLR(opt, _LOWER_LR_, _UPPER_LR_, step_size_up=_LR_STEP_, mode='triangular2', cycle_momentum=False)
    
    # * sets the device  
    _DEVICE_ = 'cuda' if torch.cuda.is_available() else 'cpu'
    util.logprint(f'_models/{_NAME_}/{_LOG_FILE_}', f'device: {_DEVICE_}')
    torch.backends.cudnn.benchmark = True
    torch.cuda.empty_cache()
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    
    # * print out parameters
    util.logprint(f'_models/{_NAME_}/{_LOG_FILE_}', f'model: {_NAME_}')
    util.logprint(f'_models/{_NAME_}/{_LOG_FILE_}', f'type: {_MODEL_TYPE_}')
    util.logprint(f'_models/{_NAME_}/{_LOG_FILE_}', f'batch-size: {_BATCH_SIZE_}')
    util.logprint(f'_models/{_NAME_}/{_LOG_FILE_}', f'pool-size: {_POOL_SIZE_}')
    util.logprint(f'_models/{_NAME_}/{_LOG_FILE_}', f'lr: {_UPPER_LR_}>{_LOWER_LR_} w/ {_LR_STEP_} step')

    # * create seed
    seed_ten = util.create_seed(_size=_SIZE_+(2*_PAD_), _dist=_SEED_DIST_, _points=_SEED_POINTS_).unsqueeze(0).to(_DEVICE_)
    util.logprint(f'_models/{_NAME_}/{_LOG_FILE_}', f'seed.shape: {list(seed_ten.shape)}')

    # * load target image
    target_img = util.load_image_as_tensor(_TARGET_IMG_)
    target_img = func.pad(target_img, (_PAD_, _PAD_, _PAD_, _PAD_), 'constant', 0)
    util.show_tensor_as_image(target_img)
    util.logprint(f'_models/{_NAME_}/{_LOG_FILE_}', f'target.shape: {list(target_img.shape)}')
    
    # * create target batch
    target_batch = target_img.clone().repeat(_BATCH_SIZE_, 1, 1, 1).to(_DEVICE_)
    
    # * create training pool
    with torch.no_grad():
        pool = seed_ten.clone().repeat(_POOL_SIZE_, 1, 1, 1)
        if model.is_steerable():
            pass
            # * randomize angles for steerable models
            # if _ANGLE_CHANNEL_ == 'RANDOMIZED' or _ANGLE_CHANNEL_ == 'SEED_DIR':
            #     for i in range(_POOL_SIZE_):
            #         rand = torch.rand(_SIZE_, _SIZE_)*np.pi*2.0
            #         pool[i, -1:] = rand
            #         if _ANGLE_CHANNEL_ == 'SEED_DIR':
            #             pool[i, -1:, _SIZE_//2+1, _SIZE_//2] = 0.0
            #             pool[i, -1:, _SIZE_//2+1, _SIZE_//2+1] = 0.0
            #             pool[i, -1:, _SIZE_//2+1, _SIZE_//2-1] = 0.0
            #             pool[i, -1:, _SIZE_//2, _SIZE_//2] = 0.0
            #             pool[i, -1:, _SIZE_//2, _SIZE_//2+1] = 0.0
            #             pool[i, -1:, _SIZE_//2, _SIZE_//2-1] = 0.0
            #             pool[i, -1:, _SIZE_//2-1, _SIZE_//2] = 0.0
            #             pool[i, -1:, _SIZE_//2-1, _SIZE_//2+1] = 0.0
            #             pool[i, -1:, _SIZE_//2-1, _SIZE_//2-1] = 0.0
    
    # * begin training 
    util.logprint(f'_models/{_NAME_}/{_LOG_FILE_}', f'starting training w/ {_EPOCHS_+1} epochs...')
    train_start = datetime.datetime.now()
    loss_log = []
    prev_lr = -np.inf
    for i in range(_EPOCHS_+1):
        with torch.no_grad():
            # * sample batch from pool
            i = len(loss_log)
            batch_idxs = np.random.choice(_POOL_SIZE_, _BATCH_SIZE_, replace=False)
            x = pool[batch_idxs]
            
            # * re-order batch based on loss
            loss_ranks = torch.argsort(util.pixel_wise_loss_function(x, target_batch, _dims=[-2, -3, -1]), descending=True)
            x = x[loss_ranks]
            
            # * re-add seed into batch
            x[:1] = seed

            # * randomize angles for steerable models
            if _IS_STEERABLE_:
                # * randomize angles for steerable models
                if _ANGLE_CHANNEL_ == 'RANDOMIZED' or _ANGLE_CHANNEL_ == 'SEED_DIR':
                    rand = torch.rand(_SIZE_, _SIZE_)*np.pi*2.0
                    x[:1, -1:] = rand
                    if _ANGLE_CHANNEL_ == 'SEED_DIR':
                        x[:1, -1:, _SIZE_//2+1, _SIZE_//2] = 0.0
                        x[:1, -1:, _SIZE_//2+1, _SIZE_//2+1] = 0.0
                        x[:1, -1:, _SIZE_//2+1, _SIZE_//2-1] = 0.0
                        x[:1, -1:, _SIZE_//2, _SIZE_//2] = 0.0
                        x[:1, -1:, _SIZE_//2, _SIZE_//2+1] = 0.0
                        x[:1, -1:, _SIZE_//2, _SIZE_//2-1] = 0.0
                        x[:1, -1:, _SIZE_//2-1, _SIZE_//2] = 0.0
                        x[:1, -1:, _SIZE_//2-1, _SIZE_//2+1] = 0.0
                        x[:1, -1:, _SIZE_//2-1, _SIZE_//2-1] = 0.0
                # * set direction of growth
                elif _ANGLE_CHANNEL_ == 'DIRECTION':
                    x[:1, -1:] = 0
                
            # * damage lowest loss in batch
            if i % _DAMG_RATE_ == 0:
                # * use random half mask
                if i % 10 == 0:
                    mask = half_mask(_SIZE_, 'rand')
                # * use random circle mask
                else:
                    radius = random.uniform(_SIZE_*0.05, _SIZE_*0.2)
                    u = random.uniform(0, 1) * _SIZE_
                    v = random.uniform(0, 1) * _SIZE_
                    mask = circle_mask(_SIZE_, radius, [u, v])
                x[-_NUM_DAMG_:] *= torch.tensor(mask).to(_DEVICE_)
                
        # * save batch before
        if i % _INFO_RATE_ == 0:
            before = x.detach().cpu()

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
            if _IS_STEERABLE_:
                overflow_loss += (x - x.clamp(-2.0, 2.0))[:, :_CHANNELS_-1].square().sum()
            else:
                overflow_loss += (x - x.clamp(-2.0, 2.0))[:, :_CHANNELS_].square().sum()
        
        # * calculate losses
        target_loss += loss_func[_LOSS_FUNC_](x, target_batch)
        target_loss /= 2.0
        diff_loss *= 10.0
        loss = target_loss + overflow_loss + diff_loss

        
        # * backward pass
        with torch.no_grad():
            loss.backward()
            # * normalize gradients 
            for p in model.parameters():
                p.grad /= (p.grad.norm()+1e-8) 
            opt.step()
            opt.zero_grad()
            lr_sched.step()
            # * re-add batch to pool
            pool[batch_idxs] = x
            loss_log.append(loss.item())
            
            # * print out info
            if i % _INFO_RATE_ == 0:
                # * show loss plot
                clear_output(True)
                pl.plot(loss_log, '.', alpha=0.1)
                pl.yscale('log')
                pl.ylim(np.min(loss_log), loss_log[0])
                pl.show()
                
                # * show batch
                after = x.detach().cpu()
                show_batch(_BATCH_SIZE_, before, after)
                
                # * print info
                print('\rstep:', i, '\tloss:', loss.item(), '\tmin-loss:', np.min(loss_log),  '\tlr:', lr_sched.get_last_lr()[0], end='')
                    
            # * save checkpoint
            if i % _SAVE_RATE_ == 0 and i != 0:
                save_model('_checkpoints', model, _NAME_+'_cp'+str(i))
                
            # * create video
            if i % _VIDEO_RATE_ == 0 and i != 0:
                vidgen(f'_videos/{_NAME_}_cp{i}.mp4', model, p=1, n_frames=256, sz=_SIZE_)
                
    # * save final model
    if _TRAIN_MODEL_:
        save_model('_models', model, _NAME_)
        vidgen(f'_videos/_final_{_NAME_}.mp4', model, n_frames=256, sz=_SIZE_)
    
if __name__ == '__main__':
    main()