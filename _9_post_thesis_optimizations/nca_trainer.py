import torch

class nca_trainer():
    def __init__(
        self, 
        _model: torch.nn.Module, 
        _optim: torch.optim.Optimizer,
        _sched: torch.optim.lr_scheduler,
        _seed:  torch.Tensor,
        _trgt:  torch.Tensor,
        _pool:  torch.Tensor,
        ):
        self.model = _model
        self.optim = _optim
        self.sched = _sched
        self.seed  = _seed
        self.trgt  = _trgt
        self.pool  = _pool
        
    def begin(
        self,
        _epochs:    int,
        _dmg_iter:  int,
        _dmg_n:     int,
        _log_file:  int,
        _info_iter: int,
        ):
        pass
        