from utility import *

from radial_automata import RadialAutomata
from decisional_nn import DecisionalNN

_NAME_ = 'cowboy_test'
_LOG_ = 'trainlog.txt'

_TARGET_FILE_ = 'cowboy.png'
_TARGET_SIZE_ = 40
_PAD_ = 12

_RADIUS_ = 1.0
_RATE_ = 0.01
_NUM_ = 16

_EPOCHS_ = 5_000
_POOL_SIZE_ = 32
_BATCH_SIZE_ = 4
_UPPER_LR_ = 5e-4
_LOWER_LR_ = 1e-5
_LR_STEP_ = 2000
_INFO_RATE_ = 100
_SAVE_RATE_ = 1000

def logprint(_path, _str):
    print (_str)
    with open(_path, 'a', encoding='utf-8') as f:
        f.write(f'{_str}\n')

def force_cudnn_initialization():
    s = 32
    dev = torch.device('cuda')
    torch.nn.functional.conv2d(torch.zeros(s, s, s, s, device=dev), torch.zeros(s, s, s, s, device=dev))

# * save model method
def save_model(_dir, _model, _name):
    # * create directory
    model_path = pathlib.Path(f'{_dir}/{_NAME_}')
    model_path.mkdir(parents=True, exist_ok=True)
    torch.save(_model.state_dict(), f'{model_path.absolute()}/{_name}.pt')
    
    # * save model parameters
    dict = {
        # * target/seed parameters
        'name': _NAME_,
        'log': _LOG_,
        'target_file': _TARGET_FILE_,
        'target_size': _TARGET_SIZE_,
        'pad': _PAD_,
        'radius': _RADIUS_,
        'rate': _RATE_,
        'num': _NUM_,
        'epochs': _EPOCHS_,
        'pool_size': _POOL_SIZE_,
        'batch_size': _BATCH_SIZE_,
        'upper_lr': _UPPER_LR_,
        'lower_lr': _LOWER_LR_,
        'lr_step': _LR_STEP_,
        'info_rate': _INFO_RATE_,
        'save_rate': _SAVE_RATE_,
    }
    json_object = json.dumps(dict, indent=4)
    with open(f'{model_path.absolute()}/{_name}_params.json', 'w') as outfile:
        outfile.write(json_object)
    logprint(f'_models/{_NAME_}/{_LOG_}', f'model [{_name}] saved to {_dir}...')

# * make directory for model files
if not os.path.exists(f'_models/{_NAME_}'):
    os.mkdir(f'_models/{_NAME_}')

# * begin logging and start program timer
logprint(f'_models/{_NAME_}/{_LOG_}', '****************')
logprint(f'_models/{_NAME_}/{_LOG_}', f'timestamp: {datetime.datetime.now()}')
logprint(f'_models/{_NAME_}/{_LOG_}', 'initializing training...')
start = datetime.datetime.now()

# * sets the device  
_DEVICE_ = 'cuda' if torch.cuda.is_available() else 'cpu'
logprint(f'_models/{_NAME_}/{_LOG_}', f'device: {_DEVICE_}')
torch.backends.cudnn.benchmark = True
torch.cuda.empty_cache()
torch.set_default_tensor_type('torch.cuda.FloatTensor')
force_cudnn_initialization()

# * create model
model = DecisionalNN(16, 19) # input: [[0-3]rgba(4), [4-15]hidden(12)] output: [[0-3]rgba(4), [4-15]hidden(12), [16-17]move(2), [18]angle(1)]
logprint(f'_models/{_NAME_}/{_LOG_}', 'training new model from scratch...')

# * create optimizer and learning-rate scheduler
opt = torch.optim.Adam(model.parameters(), _UPPER_LR_)
lr_sched = torch.optim.lr_scheduler.CyclicLR(opt, _LOWER_LR_, _UPPER_LR_, step_size_up=_LR_STEP_, mode='triangular2', cycle_momentum=False)

# * load target vox
def load_image_as_tensor(_path, _size, _resample=pil.Resampling.BICUBIC):
    img = pil.open(_path)
    img = img.resize((_size, _size), _resample)
    img = np.float32(img) / 255.0
    img[..., :3] *= img[..., 3:]
    return torch.from_numpy(img).permute(2, 0, 1)[None, ...]

# * load target
target = load_image_as_tensor('../images/'+_TARGET_FILE_, _TARGET_SIZE_)
target = torch.nn.functional.pad(target, (_PAD_, _PAD_, _PAD_, _PAD_), 'constant', 0)
logprint(f'_models/{_NAME_}/{_LOG_}', f'target_img.shape: {target.shape}')

# * create automata pool
pool = []
for i in range(_POOL_SIZE_):
    automata = RadialAutomata(_RADIUS_, _RATE_, _NUM_)
    pool.append(automata)
    
# * loss function
def loss_fn(_x, _target, _scale=1e3, _dims=[]):
    losses = []
    for automata in _x:
        image = automata.pixelize(2, 64)
        image = np.rot90(image, 1)
        image = torch.from_numpy(image.copy()).permute(2, 0, 1).unsqueeze(0)
        loss = _scale * torch.mean(torch.square(image[:, :4] - _target), _dims)
        losses.append(loss)
    return losses

# * model training
logprint(f'_models/{_NAME_}/{_LOG_}', f'starting training w/ {_EPOCHS_+1} epochs...')
train_start = datetime.datetime.now()
loss_log = []
prev_lr = -np.inf
for i in range(_EPOCHS_+1):
    with torch.no_grad():
        # * sample batch from pool
        batch_idxs = np.random.choice(_POOL_SIZE_, _BATCH_SIZE_, replace=False)
        x = []
        for j in range(len(batch_idxs)):
            x.append(pool[j])
            
        print (f'x: {x}')
            
        # * re-order batch based on loss
        losses = loss_fn(x, target)
        print (f'losses: {losses}')
        loss_ranks = torch.argsort(losses, descending=True)
        
        print (f'loss_ranks: {loss_ranks}')
        x = x[loss_ranks]
        
        print (f'x: {x}')
        
        # * re-add seed into batch
        x[:1] = RadialAutomata(_RADIUS_, _RATE_, _NUM_)
        
        # TODO * damage lowest loss in batch

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
        overflow_loss += (x - x.clamp(-2.0, 2.0))[:, :15].square().sum()

    
    # * calculate losses
    target_loss += loss_fn(x, target)
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
            logprint(f'_models/{_NAME_}/{_LOG_}', f'detected invalid loss value: {loss}')
            logprint(f'_models/{_NAME_}/{_LOG_}', f'overflow loss: {overflow_loss}, diff loss: {diff_loss}, target loss: {target_loss}')
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
            logprint(f'_models/{_NAME_}/{_LOG_}', f'[{i}/{_EPOCHS_+1}]\t {np.round(iter_per_sec, 3)}it/s\t time: {time}~{est}\t loss: {np.round(avg, 3)}>{np.round(np.min(loss_log), 3)}\t lr: {lr} {step}')
        
        # * save checkpoint
        if i % _SAVE_RATE_ == 0 and i != 0:
            save_model('_checkpoints', model, _NAME_+'_cp'+str(i))
            
# * print train time
secs = (datetime.datetime.now()-train_start).seconds
train_time = str(datetime.timedelta(seconds=secs))
logprint(f'_models/{_NAME_}/{_LOG_}', f'train time: {train_time}')

# * save loss plot
plt.plot(loss_log, '.', alpha=0.1)
plt.yscale('log')
plt.ylim(np.min(loss_log), loss_log[0])
plt.savefig(f'_models/{_NAME_}/{_NAME_}_loss_plot.png')
            
# * save final model
save_model('_models', model, _NAME_)

# * calculate elapsed time
secs = (datetime.datetime.now()-start).seconds
elapsed_time = str(datetime.timedelta(seconds=secs))
logprint(f'_models/{_NAME_}/{_LOG_}', f'elapsed time: {elapsed_time}')
logprint(f'_models/{_NAME_}/{_LOG_}', '****************')