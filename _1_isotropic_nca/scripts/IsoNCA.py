import torch

class IsoNCA(torch.nn.Module):
    def __init__(self, _name, _log_file, _channels=16, _hidden=128, _device='cuda', _model_type='ANGLE_STEER', _update_rate=0.5):
        super().__init__()
        
    def is_steerable(self):
        return self.model_type == 'ANGLE_STEER'