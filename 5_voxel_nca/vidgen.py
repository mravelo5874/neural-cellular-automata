import os
import json
import torch
import datetime
from numpy import pi

from scripts.nca.VoxelNCA import VoxelNCA as NCA
from scripts.nca import VoxelUtil as util

def main():
    # * make directory for model files
    if not os.path.exists(f'_models/{_NAME_}'):
        os.mkdir(f'_models/{_NAME_}')
        
    util.logprint(f'_models/{_NAME_}/{_LOG_FILE_}', '****************')
    util.logprint(f'_models/{_NAME_}/{_LOG_FILE_}', 'initializing video generation...')
    start = datetime.datetime.now()
    
    # * vidgen params
    _NAME_ = 'cowboy16_yawiso5'
    _DIR_ = '_models'
    _DEVICE_ = 'cuda'
    _LOG_FILE_ = 'vidlog.txt'
    
    # * set cuda as device
    util.logprint(f'_models/{_NAME_}/{_LOG_FILE_}', 'device:', _DEVICE_)
    torch.backends.cudnn.benchmark = True
    torch.cuda.empty_cache()
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    
    # * load model
    path = f'{_DIR_}/{_NAME_}/{_NAME_}.pt'
    model = NCA(_name=_NAME_, _log_file=_LOG_FILE_)
    model.load_state_dict(torch.load(path, map_location=_DEVICE_))   
    model.eval()
    util.logprint(f'_models/{_NAME_}/{_LOG_FILE_}', f'loaded model from: {path}')
    
    # * load in params for seed
    params = {}
    with open(f'{_DIR_}/{_NAME_}/{_NAME_}_params.json', 'r') as openfile:
        params = json.load(openfile)
        _SIZE_ = params['_SIZE_']
        _PAD_ = params['_PAD_']
        _SEED_POINTS_ = params['_SEED_POINTS_']
        _SEED_DIST_ = params['_SEED_DIST_']
        
    # * create seed
    seed_ten = util.create_seed(_size=_SIZE_+(2*_PAD_), _dist=_SEED_DIST_, _points=_SEED_POINTS_).unsqueeze(0).to(_DEVICE_)
    util.logprint(f'_models/{_NAME_}/{_LOG_FILE_}', f'seed.shape: {seed_ten.shape}')

    # * generate video
    s = _SIZE_+(2*_PAD_)
    curr = datetime.datetime.now().strftime("%H:%M:%S")
    util.logprint(f'_models/{_NAME_}/{_LOG_FILE_}', f'starting time: {curr}')
    util.logprint(f'_models/{_NAME_}/{_LOG_FILE_}', 'generating videos...')
    with torch.no_grad():
        # * randomize last channel
        if model.is_steerable():
            seed_ten[:1, -1:] = torch.rand(s, s, s)*pi*2.0
        model.generate_video(f'_models/{_NAME_}/vid_{_NAME_}_grow.mp4', seed_ten)
        # * randomize last channel
        if model.is_steerable():
            seed_ten[:1, -1:] = torch.rand(s, s, s)*pi*2.0
        model.regen_video(f'_models/{_NAME_}/vid_{_NAME_}_multi_regen.mp4', seed_ten, _size=s, _mask_types=['x+', 'y+', 'z+'])
        # * randomize last channel
        if model.is_steerable():
            seed_ten[:1, -1:] = torch.rand(s, s, s)*pi*2.0
        model.rotate_video(f'_models/{_NAME_}/vid_{_NAME_}_multi_rotate.mp4', seed_ten, _size=s)
        
     # * calculate elapsed time
    secs = (datetime.datetime.now()-start).seconds
    elapsed_time = str(datetime.timedelta(seconds=secs))
    util.logprint(f'_models/{_NAME_}/{_LOG_FILE_}', f'elapsed time: {elapsed_time}')
    util.logprint(f'_models/{_NAME_}/{_LOG_FILE_}', '****************')

if __name__ == '__main__':
    main()