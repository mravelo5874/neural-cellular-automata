import sys
import os
import glm
import json
import time
import torch
import pickle
import threading
import numpy as np

# * add path to '_5_voxel_nca' to sys
cwd = os.getcwd().split('\\')[:-1]
cwd = '/'.join(cwd)
sys.path.insert(1, cwd+'/_5_voxel_nca')

from scripts.nca.VoxelNCA import VoxelNCA as NCA
from scripts.nca import VoxelUtil as voxutil
from utils import Utils as utils

class NCASimulator:
    def __init__(self, _model, _engine, _device='cuda'):
        self.is_loaded = False
        self.is_running = False
        self.is_paused = False
        self.started = False
        self.device = _device
        self.engine = _engine
        self.mutex = threading.Lock()
        self.iter = 0
        
        # * setup cuda if available
        torch.backends.cudnn.benchmark = True
        if self.device == 'cuda' and torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
        else:
            self.device = 'cpu'
            
        # * load nca model
        self.load_model(_model)
        
    def load_model(self, _model):
        # * load in params for seed
        self.iter = 0
        params = {}
        with open(f'{cwd}/models/{_model}/{_model}_params.json', 'r') as openfile:
            params = json.load(openfile)
        
        # * get perception function
        pfunc = None
        ppath = f'{cwd}/models/{_model}/{_model}_perception_func.pyc'
        if os.path.exists(ppath):
            with open(ppath, 'rb') as pfile:
                pfunc = pickle.load(pfile)
                pfile.close()
            
        model = NCA(_name=params['_NAME_'], _channels=params['_CHANNELS_'], _hidden=params['_HIDDEN_'], _device=self.device, _model_type=params['_MODEL_TYPE_'], _pfunc=pfunc)
        model.load_state_dict(torch.load(f'{cwd}/models/{_model}/{_model}.pt', map_location=self.device))
        model.eval()
        self.model = model
            
        # * create seed
        _SIZE_ = params['_SIZE_']
        _PAD_ = params['_PAD_']
        _CHANNELS_ = params['_CHANNELS_']
        _USE_SPHERE_SEED_ = params['_USE_SPHERE_SEED_']
        _SEED_POINTS_ = params['_SEED_POINTS_']
        _SEED_DIST_ = params['_SEED_DIST_']
        _SEED_DIC_ = params['_SEED_DIC_']
        _SEED_HID_INFO_ = params['_SEED_HID_INFO_']
        self.size = int(_SIZE_+(3*_PAD_)*3)
        if _USE_SPHERE_SEED_:
            self.seed = voxutil.seed_3d(_size=self.size, _channels=_CHANNELS_, _points=_SEED_POINTS_, _radius=_SEED_DIST_).unsqueeze(0).to(self.device)
        else:
            self.seed = voxutil.custom_seed(_size=self.size, _channels=_CHANNELS_, _dist=_SEED_DIST_, _hidden_info=_SEED_HID_INFO_,
                                        _center=_SEED_DIC_['center'], 
                                        _plus_x=_SEED_DIC_['plus_x'], _minus_x=_SEED_DIC_['minus_x'],
                                        _plus_y=_SEED_DIC_['plus_y'], _minus_y=_SEED_DIC_['minus_y'],
                                        _plus_z=_SEED_DIC_['plus_z'], _minus_z=_SEED_DIC_['minus_z']).unsqueeze(0).to(self.device)
        # * randomize channels
        if model.isotropic_type() == 1:
                self.seed[:1, -1:] = torch.rand(self.size, self.size, self.size)*np.pi*2.0
        elif model.isotropic_type() == 3:
            self.seed[:1, -1:] = torch.rand(self.size, self.size, self.size)*np.pi*2.0
            self.seed[:1, -2:-1] = torch.rand(self.size, self.size, self.size)*np.pi*2.0
            self.seed[:1, -3:-2] = torch.rand(self.size, self.size, self.size)*np.pi*2.0
            
        # * set tensor
        self.mutex.acquire()
        self.x = self.seed.detach().clone()
        self.mutex.release()
        self.is_loaded = True
        
        name = params['_NAME_']
        print (f'loaded model: {name}')
        
    def run_thread(self, _delay):
        # * wait 3 seconds to show off seed
        time.sleep(_delay)
        self.is_running = True
        while self.is_running:
            with torch.no_grad():
                self.mutex.acquire()
                if not self.is_paused:
                    self.x = self.model(self.x)
                    self.iter += 1
                    self.started = True
                else:
                    # * dummy forward (needed so engine does not lag when paused)
                    self.model(self.x)
                self.mutex.release()
                
    def run_raycaster(self):
        while self.is_running:
            # * fire raycast from mouse pos through volume
            pos = self.engine.player.pos
            vec = self.engine.player.forward
            voxel = None
            if self.engine.SEND_RAYCASTS:
                voxel = self.raycast_volume(pos, vec)
            if voxel == None:
                # * dummy forward (needed so engine does not lag)
                self.model(self.x)
                self.engine.my_voxel = [1e12, 1e12, 1e12]
            else:
                # * dummy forward (needed so engine does not lag)
                self.model(self.x)
                self.engine.my_voxel = voxel
                
    def run(self, _delay=0.0):
        # * can only start running if not running
        if self.is_running:
            return
        # * start forward worker
        self.worker = threading.Thread(target=self.run_thread, args=[_delay], daemon=True)
        self.worker.start()
        # * start raycaster
        self.raycaster = threading.Thread(target=self.run_raycaster, daemon=True)
        self.raycaster.start()
        
    def step_forward(self):
        with torch.no_grad():
            self.mutex.acquire()
            if self.is_paused:
                self.x = self.model(self.x)
                self.iter += 1
            self.mutex.release()
            
    def rot_seed(self, _axis='z'):
        # can only rotate seed if not started
        if not self.started:
            if _axis == 'x':
                self.seed = torch.rot90(self.seed, 1, (2, 3))
            elif _axis == 'y':
                self.seed = torch.rot90(self.seed, 1, (2, 4))
            elif _axis == 'z':
                self.seed = torch.rot90(self.seed, 1, (3, 4))
            self.mutex.acquire()
            self.x = self.seed.detach().clone()
            self.mutex.release()
     
    def stop(self):
        self.is_running = False
        self.mutex.acquire()
        self.is_paused = False
        self.started = False
        self.mutex.release()
        # * join any running threads
        self.raycaster.join()
        self.worker.join()
        
    def unload(self):
        # * can only unload if loaded and running
        if self.is_loaded and self.is_running:
            self.is_loaded = False
            self.stop()
            return True
        return False
  
    def toggle_pause(self):
        # * can only pause if running
        if self.is_running:
            self.mutex.acquire()
            self.is_paused = not self.is_paused
            self.mutex.release()

    def reset(self):
        # * can only reset if running
        if self.is_running:
            self.iter = 0
            self.is_running = False
            # * join any running threads
            if self.worker:
                self.worker.join()
            if self.raycaster:
                self.raycaster.join()
            # * reset seed
            self.mutex.acquire()
            self.started = False
            self.x = self.seed.detach().clone()
            self.mutex.release()
            self.run()
        
    def get_data(self):
        # * get tensor
        self.mutex.acquire()
        data = self.x.cpu().detach().numpy()
        self.mutex.release()
        
        # convert numbers to -> np.uint8 values between 0-255
        data = data[:, :4, ...]
        data = np.clip(data, 0.0, 1.0)
        data = np.transpose(data, (0, 4, 3, 2, 1))
        _, x, y, z, rgba = data.shape
        data = data.reshape((x*y*z, rgba))*255
        data = data.astype(np.uint8)
        return data.tobytes()
    
    def raycast_volume(self, _pos, _vec):
        pos = glm.vec3(_pos[0], _pos[1], _pos[2])
        vec = glm.normalize(_vec)
        
        # * find the init voxel
        # * determine if ray starts in volume or outside volume
        
        # * find where ray intersects volume on the surface
        t_min, t_max = utils.ray_box_intersection(pos, vec, glm.vec3(-1), glm.vec3(0.999))
        
        # * exit if no intersection found
        if t_min == None or t_max == None:
            return None
        
        # * inside volume
        if utils.is_point_in_box(pos, glm.vec3(-1), glm.vec3(0.999)):
            # * normalize point to be within 0, 1
            norm_pos = (pos+1)/2
            # * multiply by size to get approximate voxel
            vox = glm.floor(norm_pos*self.size)
            
        # * outside volume
        else:
            # * normalize point to be within 0, 1
            int_pos = pos+(vec*t_min)
            norm_pos = (int_pos+1)/2
            # * multiply by size to get approximate voxel
            vox = glm.floor(norm_pos*self.size)
    
        # * run fast voxel traversal algorithm
        # * ref: https://github.com/cgyurgyik/fast-voxel-traversal-algorithm/blob/master/overview/FastVoxelTraversalOverview.md
        
        # * copy x as numpy array
        self.mutex.acquire()
        vol = np.array(self.x.cpu())
        self.mutex.release()
        
        # * get voxel alpha value only
        vol = vol.squeeze(0)[3, ...]
             
        # * init step values
        # * init t-max values
        # * init delta values
        vox_size = 2/self.size
        step_x, step_y, step_z = 0, 0, 0
        if vec.x > 0:
            step_x = 1
            t_delta_x = vox_size/_vec.x
            t_max_x = t_min + (-1.0 + vox.x * vox_size - _pos.x) / _vec.x
        elif vec.x < 0:
            step_x = -1
            t_delta_x = -vox_size/_vec.x
            t_max_x = t_min + (-1.0 + (vox.x-1) * vox_size - _pos.x) / _vec.x
        else:
            step_x = 0
            t_delta_x = t_max
            t_max_x = t_max
            
        if vec.y > 0:
            step_y = 1
            t_delta_y = vox_size/_vec.y
            t_max_y = t_min + (-1.0 + vox.y * vox_size - _pos.y) / _vec.y
        elif vec.y < 0:
            step_y = -1
            t_delta_y = -vox_size/_vec.y
            t_max_y = t_min + (-1.0 + (vox.y-1) * vox_size - _pos.y) / _vec.y
        else:
            step_y = 0
            t_delta_y = t_max
            t_max_y = t_max
            
        if vec.z > 0:
            step_z = 1
            t_delta_z = vox_size/_vec.z
            t_max_z = t_min + (-1.0 + vox.z * vox_size - _pos.z) / _vec.z
        elif vec.z < 0:
            step_z = -1
            t_delta_z = -vox_size/_vec.z
            t_max_z = t_min + (-1.0 + (vox.z-1) * vox_size - _pos.z) / _vec.z
        else:
            step_z = 0
            t_delta_z = t_max
            t_max_z = t_max

        # * run algorithm main loop
        while True:
            if t_max_x < t_max_y:
                if t_max_x < t_max_z:
                    vox.x += step_x
                    if vox.x < 0 or vox.x >= self.size:
                        return None
                    t_max_x += t_delta_x
                else:
                    vox.z += step_z
                    if vox.z < 0 or vox.z >= self.size:
                        return None
                    t_max_z += t_delta_z
            else:
                if t_max_y < t_max_z:
                    vox.y += step_y
                    if vox.y < 0 or vox.y >= self.size:
                        return None
                    t_max_y += t_delta_y
                else:
                    vox.z += step_z
                    if vox.z < 0 or vox.z >= self.size:
                        return None
                    t_max_z += t_delta_z
            # * check for out of bounds
            if (vox.x < 0 or vox.x >= self.size or
                vox.y < 0 or vox.y >= self.size or
                vox.z < 0 or vox.z >= self.size):
                return None
            # * check if voxel is "alive"
            if vol[int(vox.x), int(vox.y), int(vox.z)] > 0.1:
                break
        return vox
    
    def erase_sphere(self, _pos, _radius):
        X, Y, Z = np.ogrid[:self.size, :self.size, :self.size]
        dist_from_center = np.sqrt((X-_pos[0])**2 + (Y-_pos[1])**2 + (Z-_pos[2])**2)
        mask = torch.tensor(dist_from_center >= _radius)
        mask = mask[None, None, ...]
        
        self.mutex.acquire()
        self.x = self.x*mask
        self.mutex.release()
        
    def load_custom(self, _num):
        # can only load custom if not started
        if not self.started:
            print (f'loading custom seed {_num}...')
            
            # * copy seed in each quadrant with a different rotation
            if _num == 0:
                full=self.seed.shape[2]
                half=full//2
                q=half//2
                d=q//2
     
                clone = self.seed.detach().clone()[:, :, half-d:half+d, half-d:half+d, half-d:half+d]
                self.seed = torch.zeros_like(self.seed)
                
                # * bottom half
                self.seed[:, :,         q-d:q+d,            q-d:q+d,            q-d:q+d] = torch.rot90(clone, 1, (4, 3))
                self.seed[:, :, (q*3)-d:(q*3)+d,            q-d:q+d,            q-d:q+d] = torch.rot90(clone, 2, (4, 3))
                self.seed[:, :, (q*3)-d:(q*3)+d,    (q*3)-d:(q*3)+d,            q-d:q+d] = torch.rot90(clone, 3, (4, 3))
                self.seed[:, :,         q-d:q+d,    (q*3)-d:(q*3)+d,            q-d:q+d] = torch.rot90(clone, 4, (4, 3))
                # * top half
                self.seed[:, :,         q-d:q+d,            q-d:q+d,    (q*3)-d:(q*3)+d] = torch.rot90(clone, 3, (2, 3))
                self.seed[:, :, (q*3)-d:(q*3)+d,            q-d:q+d,    (q*3)-d:(q*3)+d] = torch.rot90(clone, 4, (2, 3))
                self.seed[:, :, (q*3)-d:(q*3)+d,    (q*3)-d:(q*3)+d,    (q*3)-d:(q*3)+d] = torch.rot90(clone, 1, (2, 3))
                self.seed[:, :,         q-d:q+d,    (q*3)-d:(q*3)+d,    (q*3)-d:(q*3)+d] = torch.rot90(clone, 2, (2, 3))
                
                # * randomize channels
                if self.model.isotropic_type() == 1:
                        self.seed[:1, -1:] = torch.rand(self.size, self.size, self.size)*np.pi*2.0
                elif self.model.isotropic_type() == 3:
                    self.seed[:1, -1:] = torch.rand(self.size, self.size, self.size)*np.pi*2.0
                    self.seed[:1, -2:-1] = torch.rand(self.size, self.size, self.size)*np.pi*2.0
                    self.seed[:1, -3:-2] = torch.rand(self.size, self.size, self.size)*np.pi*2.0
                
                self.mutex.acquire()
                self.x = self.seed.detach().clone()
                self.mutex.release()
            
            # * four instances each facing a unique cardinal direction
            if _num == 1:
                self.seed = torch.zeros_like(self.seed)
                # shape: [1, 16, 32, 32, 32]
                
                full=self.seed.shape[2]
                half=full//2
                chn=self.seed.shape[1]
                rad=10
                dist=2
                
                # red
                # * top half
                self.seed[:, 0, half+rad+dist, half+rad, half+rad] = 1.0 
                self.seed[:, 0, half-rad-dist, half-rad, half+rad] = 1.0
                # * bot half
                self.seed[:, 0, half-rad, half+rad-dist, half-rad] = 1.0 
                self.seed[:, 0, half+rad, half-rad+dist, half-rad] = 1.0
                
                
                # green
                # * top half
                self.seed[:, 1, half+rad-dist, half+rad, half+rad] = 1.0
                self.seed[:, 1, half-rad+dist, half-rad, half+rad] = 1.0
                # * bot half
                self.seed[:, 1, half-rad, half+rad+dist, half-rad] = 1.0
                self.seed[:, 1, half+rad, half-rad-dist, half-rad] = 1.0
            
                # alpha + hidden channels
                # * top half
                self.seed[:, 3:chn, half+rad+dist, half+rad, half+rad] = 1.0
                self.seed[:, 3:chn, half-rad-dist, half-rad, half+rad] = 1.0
                self.seed[:, 3:chn, half+rad-dist, half+rad, half+rad] = 1.0
                self.seed[:, 3:chn, half-rad+dist, half-rad, half+rad] = 1.0
                # * bot half
                self.seed[:, 3:chn, half-rad, half+rad-dist, half-rad] = 1.0 
                self.seed[:, 3:chn, half+rad, half-rad+dist, half-rad] = 1.0
                self.seed[:, 3:chn, half-rad, half+rad+dist, half-rad] = 1.0
                self.seed[:, 3:chn, half+rad, half-rad-dist, half-rad] = 1.0
                
                # * random last state
                self.seed[: -1:] = torch.rand(full, full, full)*np.pi*2.0
                        
                self.mutex.acquire()
                self.x = self.seed.detach().clone()
                self.mutex.release()
            
            # * pi/2 diagonal
            if _num == 2:
                self.seed = torch.zeros_like(self.seed)
                # shape: [1, 16, 32, 32, 32]
                
                full=self.seed.shape[2]
                half=full//2
                chn=self.seed.shape[1]
                rad=3
                
                # red
                self.seed[:, 0, half+rad, half+rad, half] = 1.0 
                # green
                self.seed[:, 1, half, half, half] = 1.0
                # alpha + hidden channels
                self.seed[:, 3:chn, half+rad, half+rad, half] = 1.0
                self.seed[:, 3:chn, half, half, half] = 1.0

                # * random last state
                self.seed[: -1:] = torch.rand(full, full, full)*np.pi*2.0
                        
                self.mutex.acquire()
                self.x = self.seed.detach().clone()
                self.mutex.release()
            
            # * eight rotations from 0 to 7/4 pi
            if _num == 3:
                
                full=self.seed.shape[2]
                half=full//2
                q=half//2
                d=q//2
                
                # * gather init seed cells
                clone = self.seed.detach().clone()[:, :, half-d:half+d, half-d:half+d, half-d:half+d]
                self.seed = torch.zeros_like(self.seed)
                
                chn = clone.shape[1]
                sz = clone.shape[-1]
                
                init_seeds = []
                s = clone.shape[-1]
                for x in range(s):
                    for y in range(s):
                        for z in range(s):
                            if clone[:, 3, x, y, z] > 0:
                                init_seeds.append((x, y, z))
                                
                # * top half              
                # * seed 0 -> 0
                self.seed[:, :,         q-d:q+d,            q-d:q+d,    (q*3)-d:(q*3)+d] = clone
                
                # * seed 1 -> 1/4 pi
                cloney = torch.zeros_like(clone)
                for i in range(len(init_seeds)):
                    x, y, z = init_seeds[i]
                    cell = clone[:, :, x, y, z]
                    p = utils.rotate_voxel(sz, np.array([x, y, z], dtype=float), 0, 0, np.pi*(1/4))
                    cloney[:, :, int(p[0]), int(p[1]), int(p[2])] = cell
                self.seed[:, :, (q*3)-d:(q*3)+d,            q-d:q+d,    (q*3)-d:(q*3)+d] = cloney
                
                # * seed 2 -> 1/2 pi
                cloney = torch.zeros_like(clone)
                for i in range(len(init_seeds)):
                    x, y, z = init_seeds[i]
                    cell = clone[:, :, x, y, z]
                    p = utils.rotate_voxel(sz, np.array([x, y, z], dtype=float), 0, 0, np.pi*(1/2))
                    cloney[:, :, int(p[0]), int(p[1]), int(p[2])] = cell
                self.seed[:, :, (q*3)-d:(q*3)+d,    (q*3)-d:(q*3)+d,    (q*3)-d:(q*3)+d] = cloney
                
                # * seed 3 -> 3/4 pi
                cloney = torch.zeros_like(clone)
                for i in range(len(init_seeds)):
                    x, y, z = init_seeds[i]
                    cell = clone[:, :, x, y, z]
                    p = utils.rotate_voxel(sz, np.array([x, y, z], dtype=float), 0, 0, np.pi*(3/4))
                    cloney[:, :, int(p[0]), int(p[1]), int(p[2])] = cell
                    self.seed[:, :,         q-d:q+d,    (q*3)-d:(q*3)+d,    (q*3)-d:(q*3)+d] = cloney
                
                # # * bottom half
                # * seed 4 -> pi
                cloney = torch.zeros_like(clone)
                for i in range(len(init_seeds)):
                    x, y, z = init_seeds[i]
                    cell = clone[:, :, x, y, z]
                    p = utils.rotate_voxel(sz, np.array([x, y, z], dtype=float), 0, 0, np.pi*(1/1))
                    cloney[:, :, int(p[0]), int(p[1]), int(p[2])] = cell
                    self.seed[:, :,         q-d:q+d,            q-d:q+d,            q-d:q+d] = cloney
                
                # * seed 5 -> 5/4 pi
                cloney = torch.zeros_like(clone)
                for i in range(len(init_seeds)):
                    x, y, z = init_seeds[i]
                    cell = clone[:, :, x, y, z]
                    p = utils.rotate_voxel(sz, np.array([x, y, z], dtype=float), 0, 0, np.pi*(5/4))
                    cloney[:, :, int(p[0]), int(p[1]), int(p[2])] = cell
                    self.seed[:, :, (q*3)-d:(q*3)+d,            q-d:q+d,            q-d:q+d] = cloney
                    
                # * seed 6 -> 3/2 pi
                cloney = torch.zeros_like(clone)
                for i in range(len(init_seeds)):
                    x, y, z = init_seeds[i]
                    cell = clone[:, :, x, y, z]
                    p = utils.rotate_voxel(sz, np.array([x, y, z], dtype=float), 0, 0, np.pi*(3/2))
                    cloney[:, :, int(p[0]), int(p[1]), int(p[2])] = cell
                    self.seed[:, :, (q*3)-d:(q*3)+d,    (q*3)-d:(q*3)+d,            q-d:q+d] = cloney
                    
                # * seed 7 -> 7/4 pi
                cloney = torch.zeros_like(clone)
                for i in range(len(init_seeds)):
                    x, y, z = init_seeds[i]
                    cell = clone[:, :, x, y, z]
                    p = utils.rotate_voxel(sz, np.array([x, y, z], dtype=float), 0, 0, np.pi*(7/4))
                    cloney[:, :, int(p[0]), int(p[1]), int(p[2])] = cell    
                    self.seed[:, :,         q-d:q+d,    (q*3)-d:(q*3)+d,            q-d:q+d] = cloney
                
                # * randomize channels
                if self.model.isotropic_type() == 1:
                        self.seed[:1, -1:] = torch.rand(self.size, self.size, self.size)*np.pi*2.0
                elif self.model.isotropic_type() == 3:
                    self.seed[:1, -1:] = torch.rand(self.size, self.size, self.size)*np.pi*2.0
                    self.seed[:1, -2:-1] = torch.rand(self.size, self.size, self.size)*np.pi*2.0
                    self.seed[:1, -3:-2] = torch.rand(self.size, self.size, self.size)*np.pi*2.0
                    
                self.mutex.acquire()
                self.x = self.seed.detach().clone()
                self.mutex.release()
                
                
                