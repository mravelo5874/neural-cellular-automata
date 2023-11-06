import json
import torch
import datetime

from scripts.nca.VoxelNCA import VoxelNCA as NCA
from scripts.nca import VoxelUtil as util

def main():
    print ('****************')
    print ('initializing video generation...')
    start = datetime.datetime.now()
    
    # * vidgen params
    _NAME_ = 'earth_aniso2'
    _DIR_ = '_models'
    _DEVICE_ = 'cuda'
    
    # * set cuda as device
    print ('device:', _DEVICE_)
    torch.backends.cudnn.benchmark = True
    torch.cuda.empty_cache()
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    
    # * load model
    path = f'{_DIR_}/{_NAME_}/{_NAME_}.pt'
    model = NCA()
    model.load_state_dict(torch.load(path, map_location=_DEVICE_))   
    model.eval()
    print (f'loaded model from: {path}')
    
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
    print (f'seed.shape: {seed_ten.shape}')

    # * generate video
    print ('generating videos...')
    with torch.no_grad():
        model.generate_video(f'_models/{_NAME_}/vid_{_NAME_}_grow.mp4', seed_ten)
        model.regen_video(f'_models/{_NAME_}/vid_{_NAME_}_multi_regen.mp4', seed_ten, _size=_SIZE_+(2*_PAD_), _mask_types=['x+', 'y+', 'z+'])
        model.rotate_video(f'_models/{_NAME_}/vid_{_NAME_}_multi_rotate.mp4', seed_ten, _size=_SIZE_+(2*_PAD_))
        
     # * calculate elapsed time
    secs = (datetime.datetime.now()-start).seconds
    elapsed_time = str(datetime.timedelta(seconds=secs))
    print (f'elapsed time: {elapsed_time}')
    print ('****************')

if __name__ == '__main__':
    main()