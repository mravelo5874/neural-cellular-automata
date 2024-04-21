import os
import json
import torch
import pickle
import numpy as np
import torch.nn.functional as func

from scripts.nca.VoxelNCA import VoxelNCA as NCA
from scripts.nca import VoxelUtil as voxutil
from scripts.vox.Vox import Vox

_MODEL_ = 'cowboy16_isoRmat_v0'
_DEVICE_ = 'cuda'
_OBJ_DIR_ = f'../obj/{_MODEL_}_regen_thesis'

_USE_DELTA_ = False
_DELTA_ITER_ = 10
_MAX_ITER_ = 60

_ITER_LIST_ = [0, 5, 10, 15, 20, 30, 50, 75, 100, 200, 500]

_REGEN_ = True
_REGEN_START_ = 200

_ISO_DEMO_ = False

def main():
    # * setup cuda if available
    torch.backends.cudnn.benchmark = True
    if _DEVICE_ == 'cuda' and torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
        
    print (f'model: {_MODEL_}')    
    
    params = {}
    with open(f'_models/{_MODEL_}/{_MODEL_}_params.json', 'r') as openfile:
        params = json.load(openfile)
    
    # * get perception function
    pfunc = None
    ppath = f'_models/{_MODEL_}/{_MODEL_}_perception_func.pyc'
    if os.path.exists(ppath):
        with open(ppath, 'rb') as pfile:
            pfunc = pickle.load(pfile)
            pfile.close()
        
    model = NCA(_name=params['_NAME_'], _channels=params['_CHANNELS_'], _hidden=params['_HIDDEN_'], _device=_DEVICE_, _model_type=params['_MODEL_TYPE_'], _pfunc=pfunc)
    model.load_state_dict(torch.load(f'_models/{_MODEL_}/{_MODEL_}.pt', map_location=_DEVICE_))
    model.eval()
        
    # * create seed
    _SIZE_ = params['_SIZE_']
    _PAD_ = params['_PAD_']
    _CHANNELS_ = params['_CHANNELS_']
    _USE_SPHERE_SEED_ = params['_USE_SPHERE_SEED_']
    _SEED_POINTS_ = params['_SEED_POINTS_']
    _SEED_DIST_ = params['_SEED_DIST_']
    _SEED_DIC_ = params['_SEED_DIC_']
    _SEED_HID_INFO_ = params['_SEED_HID_INFO_']
    _TARGET_VOX_ = params['_TARGET_VOX_']
    
    
    # place two seeds above and below at different angles
    if _ISO_DEMO_:     
        size = int(_SIZE_+(2*_PAD_))
        if _USE_SPHERE_SEED_:
            seed = voxutil.seed_3d(_size=size, _channels=_CHANNELS_, _points=_SEED_POINTS_, _radius=_SEED_DIST_).unsqueeze(0).to(_DEVICE_)
        else:
            seed = voxutil.custom_seed(_size=size, _channels=_CHANNELS_, _dist=_SEED_DIST_, _hidden_info=_SEED_HID_INFO_,
                                        _center=_SEED_DIC_['center'], 
                                        _plus_x=_SEED_DIC_['plus_x'], _minus_x=_SEED_DIC_['minus_x'],
                                        _plus_y=_SEED_DIC_['plus_y'], _minus_y=_SEED_DIC_['minus_y'],
                                        _plus_z=_SEED_DIC_['plus_z'], _minus_z=_SEED_DIC_['minus_z']).unsqueeze(0).to(_DEVICE_)
        
        half=size//2
        q=half//2
        d=q//2
        size *= 2
        size = int(size)
        
        clone = seed.detach().clone()[:, :, half-q:half+q, half-q:half+q, half-q:half+q]
        seed = torch.zeros([1, _CHANNELS_, size, size, size])
        
        h=size//2
        
        # * top half
        seed[:, :, h-q:h+q, h-q:h+q, h+d:h+q+q+d] = torch.rot90(clone, 2, (2, 3))
        # * bottom half
        seed[:, :, h-q:h+q, h-q:h+q, h-q-q-d:h-d] = torch.rot90(clone, 1, (2, 3))
        
    else:
        size = int(_SIZE_+(2*_PAD_))
        if _USE_SPHERE_SEED_:
            seed = voxutil.seed_3d(_size=size, _channels=_CHANNELS_, _points=_SEED_POINTS_, _radius=_SEED_DIST_).unsqueeze(0).to(_DEVICE_)
        else:
            seed = voxutil.custom_seed(_size=size, _channels=_CHANNELS_, _dist=_SEED_DIST_, _hidden_info=_SEED_HID_INFO_,
                                        _center=_SEED_DIC_['center'], 
                                        _plus_x=_SEED_DIC_['plus_x'], _minus_x=_SEED_DIC_['minus_x'],
                                        _plus_y=_SEED_DIC_['plus_y'], _minus_y=_SEED_DIC_['minus_y'],
                                        _plus_z=_SEED_DIC_['plus_z'], _minus_z=_SEED_DIC_['minus_z']).unsqueeze(0).to(_DEVICE_)   

    # * randomize channels
    if model.isotropic_type() == 1:
            seed[:1, -1:] = torch.rand(size, size, size)*np.pi*2.0
    elif model.isotropic_type() == 3:
        seed[:1, -1:] = torch.rand(size, size, size)*np.pi*2.0
        seed[:1, -2:-1] = torch.rand(size, size, size)*np.pi*2.0
        seed[:1, -3:-2] = torch.rand(size, size, size)*np.pi*2.0
        
    # Vox().load_from_tensor(seed).render(_show_grid=True)
        
    # * set tensor
    x = seed.detach().clone()
    
    # * create obj directory
    if not os.path.exists(_OBJ_DIR_):
        os.mkdir(_OBJ_DIR_)
        
    # * set _MAX_ITER_ if not _USE_DELTA_
    if not _USE_DELTA_:
        _MAX_ITER_ = _ITER_LIST_[-1]
    
    # * save target vox
    print (f'target vox: {_TARGET_VOX_}')
    if _TARGET_VOX_.endswith('vox'):
        target = Vox().load_from_file(_TARGET_VOX_).numpy()
    elif _TARGET_VOX_.endswith('npy'):
        with open(_TARGET_VOX_, 'rb') as f:
            target = np.load(f)
    target = target.squeeze(0)
    Vox().load_from_array(target).save_to_obj(_name=f'target_obj', _dir=_OBJ_DIR_)
    
    # * save tensor every _DELTA_ITER_ iterations
    with torch.no_grad():
        
        if not _REGEN_:
            for i in range(_MAX_ITER_+1):
                if _USE_DELTA_:
                    if i % _DELTA_ITER_ == 0:
                        Vox().load_from_tensor(x).save_to_obj(_name=f'iter_{i}', _dir=_OBJ_DIR_)
                        print (f'saving .obj for iteration {i}...')
                        Vox().save_nca_state(x.cpu().detach(), _name=f'iter_{i}', _dir=_OBJ_DIR_)
                else:
                    if i in _ITER_LIST_:
                        Vox().load_from_tensor(x).save_to_obj(_name=f'iter_{i}', _dir=_OBJ_DIR_)
                        print (f'saving .obj for iteration {i}...')
                        Vox().save_nca_state(x.cpu().detach(), _name=f'iter_{i}', _dir=_OBJ_DIR_)     
                x = model(x)
        else:
            # * get model to regen_start iteration
            for i in range(_REGEN_START_):
                x = model(x)
                
            Vox().load_from_tensor(x).save_to_obj(_name=f'init_{_REGEN_START_}', _dir=_OBJ_DIR_)
            print (f'saving .obj for iteration {_REGEN_START_}...')
            Vox().save_nca_state(x.cpu().detach(), _name=f'init_{_REGEN_START_}', _dir=_OBJ_DIR_)
                
            # * apply cellular damage
            mask = torch.tensor(voxutil.half_volume_mask(size, 'x+')).to(_DEVICE_)
            mask = mask[None, None, ...]
            x *= mask
            
            # * randomize angles for steerable models
            if model.isotropic_type() == 1:
                inv_mask = ~mask
                x[:, -1:] += torch.rand(size, size, size)*np.pi*2.0*inv_mask
            elif model.isotropic_type() == 3:
                inv_mask = ~mask
                x[:, -1:] += torch.rand(size, size, size)*np.pi*2.0*inv_mask
                x[:, -2:-1] += torch.rand(size, size, size)*np.pi*2.0*inv_mask
                x[:, -3:-2] += torch.rand(size, size, size)*np.pi*2.0*inv_mask
                
            for i in range(_MAX_ITER_+1):
                if _USE_DELTA_:
                    if i % _DELTA_ITER_ == 0:
                        Vox().load_from_tensor(x).save_to_obj(_name=f'regen_iter_{i}', _dir=_OBJ_DIR_)
                        print (f'saving .obj for iteration {i}...')
                        Vox().save_nca_state(x.cpu().detach(), _name=f'regen_iter_{i}', _dir=_OBJ_DIR_)
                else:
                    if i in _ITER_LIST_:
                        Vox().load_from_tensor(x).save_to_obj(_name=f'regen_iter_{i}', _dir=_OBJ_DIR_)
                        print (f'saving .obj for iteration {i}...')
                        Vox().save_nca_state(x.cpu().detach(), _name=f'regen_iter_{i}', _dir=_OBJ_DIR_)
                        
                x = model(x)
                
    # compute L2 loss
    target = torch.tensor(target, dtype=torch.float32).unsqueeze(0)
    target = func.pad(target, (_PAD_, _PAD_, _PAD_, _PAD_, _PAD_, _PAD_), 'constant')
    print (f'x.shape: {x.shape}')
    print (f'target.shape: {target.shape}')
    l2_loss = torch.mean(torch.square(x[:, :4] - target))
    print (f'l2 loss: {l2_loss}')
    
if __name__ == '__main__':
    main()