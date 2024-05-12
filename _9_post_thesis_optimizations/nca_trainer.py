import torch
import datetime
import matplotlib.pylab as pl
# * custom imports
from nca_util import *

class nca_trainer():
    def __init__(
        self, 
        _model: torch.nn.Module, 
        _optim: torch.optim.Optimizer,
        _sched: torch.optim.lr_scheduler,
        _seed:  torch.Tensor,
        _trgt:  torch.Tensor,
        _pool:  torch.Tensor,
        _nca_params: dict,
        _isotype: int,
        _gpu_id: int
        ):
        self.model = _model
        self.optim = _optim
        self.sched = _sched
        self.seed  = _seed
        self.trgt  = _trgt
        self.pool  = _pool
        self.nca_params = _nca_params
        self.isotype = _isotype
        self.gpu_id = _gpu_id
        
    def begin(self):
        # * regularly used params
        name = self.nca_params['_NAME_']
        logf = self.nca_params['_LOG_FILE_']
        epochs = self.nca_params['_EPOCHS_']
        size = self.nca_params['_SIZE_']+(2*self.nca_params['_PAD_'])
        ndamg = self.nca_params['_NUM_DAMG_']
        chnls = self.nca_params['_CHANNELS_']
        info = self.nca_params['_INFO_RATE_']
        save = self.nca_params['_SAVE_RATE_']
        
        # * begin logging and start program timer
        start = datetime.datetime.now()
        logprint(f'models/{name}/{logf}', '****************')
        logprint(f'models/{name}/{logf}', f'timestamp: {datetime.datetime.now()}')
        logprint(f'models/{name}/{logf}', f'starting training w/ {epochs+1} epochs...')

        loss_log = []
        prev_lr = -np.inf
        for i in range(epochs+1):
            with torch.no_grad():
                # * sample batch from pool
                batch_idxs = np.random.choice(self.nca_params['_POOL_SIZE_'], self.nca_params['_BATCH_SIZE_'], replace=False)
                x = self.pool[batch_idxs]
                
                # * re-order batch based on loss
                loss_ranks = torch.argsort(voxel_wise_loss_function(x, self.trgt, _dims=[-1, -2, -3, -4]), descending=True)
                x = x[loss_ranks]
                
                # * re-add seed into batch
                x[:1] = self.seed
                # * randomize last channel
                if self.isotype == 1:
                    x[:1, -1:] = torch.rand(size, size, size)*np.pi*2.0
                elif self.isotype == 3:
                    x[:1, -1:] = torch.rand(size, size, size)*np.pi*2.0
                    x[:1, -2:-1] = torch.rand(size, size, size)*np.pi*2.0
                    x[:1, -3:-2] = torch.rand(size, size, size)*np.pi*2.0
            
                # * damage lowest loss in batch
                if i % self.nca_params['_DAMG_RATE_'] == 0:
                    mask = torch.tensor(half_volume_mask(size, 'rand'))
                    # * apply mask
                    x[-ndamg:] *= mask
                    # * randomize angles for steerable models
                    if self.isotype == 1:
                        inv_mask = ~mask
                        x[-ndamg:, -1:] += torch.rand(size, size, size)*np.pi*2.0*inv_mask
                    elif self.isotype == 3:
                        inv_mask = ~mask
                        x[-ndamg:, -1:] += torch.rand(size, size, size)*np.pi*2.0*inv_mask
                        x[-ndamg:, -2:-1] += torch.rand(size, size, size)*np.pi*2.0*inv_mask
                        x[-ndamg:, -3:-2] += torch.rand(size, size, size)*np.pi*2.0*inv_mask

            # * different loss values
            overflow_loss = 0.0
            diff_loss = 0.0
            target_loss = 0.0
            
            # * forward pass
            num_steps = np.random.randint(64, 96)
            for _ in range(num_steps):
                prev_x = x
                x = self.model(x)
                diff_loss += (x - prev_x).abs().mean()
                if self.isotype == 1:
                    overflow_loss += (x - x.clamp(-2.0, 2.0))[:, :chnls-1].square().sum()
                elif self.isotype == 3:
                    overflow_loss += (x - x.clamp(-2.0, 2.0))[:, :chnls-3].square().sum()
                else:
                    overflow_loss += (x - x.clamp(-2.0, 2.0))[:, :chnls].square().sum()
            
            # * calculate losses
            target_loss += voxel_wise_loss_function(x, self.trgt)
            target_loss /= 2.0
            diff_loss *= 10.0
            loss = target_loss + overflow_loss + diff_loss
            
            # * backward pass
            with torch.no_grad():
                loss.backward()
                # * normalize gradients 
                for p in self.model.parameters():
                    p.grad /= (p.grad.norm()+1e-5)

                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5) # maybe? : 
                self.optim.step()
                self.optim.zero_grad()
                self.sched.step()
                # * re-add batch to pool
                self.pool[batch_idxs] = x
                # * correctly add to loss log
                _loss = loss.item()
                if torch.isnan(loss) or torch.isinf(loss) or torch.isneginf(loss):
                    pass
                else:
                    loss_log.append(_loss)
                                    
                # * detect invalid loss values :(
                if torch.isnan(loss) or torch.isinf(loss) or torch.isneginf(loss):
                    logprint(f'models/{name}/{logf}', f'detected invalid loss value: {loss}')
                    logprint(f'models/{name}/{logf}', f'overflow loss: {overflow_loss}, diff loss: {diff_loss}, target loss: {target_loss}')
                    raise ValueError
                
                # * print info
                if i % info == 0 and i!= 0:
                    secs = (datetime.datetime.now()-start).seconds
                    time = str(datetime.timedelta(seconds=secs))
                    iter_per_sec = float(i)/float(secs)
                    est_time_sec = int((epochs-i)*(1/iter_per_sec))
                    est = str(datetime.timedelta(seconds=est_time_sec))
                    avg = sum(loss_log[-info:])/float(info)
                    lr = np.round(self.sched.get_last_lr()[0], 8)
                    step = '▲'
                    if prev_lr > lr:
                        step = '▼'
                    prev_lr = lr
                    logprint(f'models/{name}/{logf}', f'[{i}/{epochs+1}]\t {np.round(iter_per_sec, 3)}it/s\t time: {time}~{est}\t loss: {np.round(avg, 3)}>{np.round(np.min(loss_log), 3)}\t lr: {lr} {step}')
                
                # * save checkpoint
                if i % save == 0 and i != 0 and self.gpu_id == 0:
                    self.model.save('models/checkpoints', name+'_cp'+str(i), self.nca_params)
                    logprint(f'models/{name}/{logf}', f'model [{name}] saved to checkpoints...')
        
        # * save loss plot
        pl.plot(loss_log, '.', alpha=0.1)
        pl.yscale('log')
        pl.ylim(np.min(loss_log), loss_log[0])
        pl.savefig(f'models/{name}/{name}_loss_plot.png')
                    
        # * save final model
        self.model.save('models', name+'_final', self.nca_params)
        
        # * calculate elapsed time
        secs = (datetime.datetime.now()-start).seconds
        elapsed_time = str(datetime.timedelta(seconds=secs))
        logprint(f'models/{name}/{logf}', f'elapsed time: {elapsed_time}')
        logprint(f'models/{name}/{logf}', '****************')