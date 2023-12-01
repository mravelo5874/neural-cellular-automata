import os
import json
import torch
import datetime
from numpy import pi

from scripts.nca.VoxelNCA import VoxelNCA as NCA
from scripts.nca import VoxelUtil as voxutil

# * vidgen params
_NAME_ = 'cowboy16_quat_2'
_DIR_ = '_models'
_DEVICE_ = 'cuda'
_LOG_FILE_ = 'vidlog.txt'

def main():
    # * make directory for model files
    if not os.path.exists(f'_models/{_NAME_}'):
        os.mkdir(f'_models/{_NAME_}')
    
    # * begin logging and start program timer
    voxutil.logprint(f'_models/{_NAME_}/{_LOG_FILE_}', '****************')
    voxutil.logprint(f'_models/{_NAME_}/{_LOG_FILE_}', f'timestamp: {datetime.datetime.now()}')
    voxutil.logprint(f'_models/{_NAME_}/{_LOG_FILE_}', 'initializing video generation...')
    start = datetime.datetime.now()
    
    # * set cuda as device
    voxutil.logprint(f'_models/{_NAME_}/{_LOG_FILE_}', f'device: {_DEVICE_}')
    torch.backends.cudnn.benchmark = True
    torch.cuda.empty_cache()
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    
    # * load in params for seed
    params = {}
    with open(f'{_DIR_}/{_NAME_}/{_NAME_}_params.json', 'r') as openfile:
        params = json.load(openfile)
        _SIZE_ = params['_SIZE_']
        _PAD_ = params['_PAD_']
        _SEED_DIST_ = params['_SEED_DIST_']
        _SEED_DIC_ = params['_SEED_DIC_']
        
    # * load model
    path = f'{_DIR_}/{_NAME_}/{_NAME_}.pt'
    model = NCA(_name=_NAME_, _log_file=_LOG_FILE_, _channels=params['_CHANNELS_'], _model_type=params['_MODEL_TYPE_'])
    model.load_state_dict(torch.load(path, map_location=_DEVICE_))   
    model.eval()
    voxutil.logprint(f'_models/{_NAME_}/{_LOG_FILE_}', f'loaded model from: {path}')
        
    # * create seed
    PAD_SIZE = _SIZE_+(2*_PAD_)
    seed_ten = voxutil.custom_seed(_size=PAD_SIZE, _channels=params['_CHANNELS_'], _dist=_SEED_DIST_, _center=_SEED_DIC_['center'], 
                                _plus_x=_SEED_DIC_['plus_x'], _minus_x=_SEED_DIC_['minus_x'],
                                _plus_y=_SEED_DIC_['plus_y'], _minus_y=_SEED_DIC_['minus_y'],
                                _plus_z=_SEED_DIC_['plus_z'], _minus_z=_SEED_DIC_['minus_z']).unsqueeze(0).to(_DEVICE_)
    voxutil.logprint(f'_models/{_NAME_}/{_LOG_FILE_}', f'seed.shape: {seed_ten.shape}')

    # * generate video
    curr = datetime.datetime.now().strftime("%H:%M:%S")
    voxutil.logprint(f'_models/{_NAME_}/{_LOG_FILE_}', f'starting time: {curr}')
    voxutil.logprint(f'_models/{_NAME_}/{_LOG_FILE_}', 'generating videos...')
    with torch.no_grad():
        # model.generate_video(f'_models/{_NAME_}/vidgen_grow.mp4', seed_ten, _size=s)
        # model.regen_video(f'_models/{_NAME_}/vidgen_multi_regen.mp4', seed_ten, _size=s, _mask_types=['x+', 'y+', 'z+'])
        model.rotate_yawiso_video(f'_models/{_NAME_}/vidgen_multi_rotate.mp4', seed_ten, _size=PAD_SIZE, _show_grid=True)
        
     # * calculate elapsed time
    secs = (datetime.datetime.now()-start).seconds
    elapsed_time = str(datetime.timedelta(seconds=secs))
    voxutil.logprint(f'_models/{_NAME_}/{_LOG_FILE_}', f'elapsed time: {elapsed_time}')
    voxutil.logprint(f'_models/{_NAME_}/{_LOG_FILE_}', '****************')

if __name__ == '__main__':
    main()