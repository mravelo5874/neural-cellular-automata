import git, os
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import to_hex
from scripts.Video import VideoWriter, zoom
import scripts.vox.VoxParser as parser

class Vox(object):
    def __init__(self):
        self.name = 'unnamed_vox'
        self.voxels = None
        self.repo_root = self.get_git_root()
    
    # * find the root of the current repository
    def get_git_root(self):
        path = os.getcwd()
        git_repo = git.Repo(path, search_parent_directories=True)
        git_root = git_repo.git.rev_parse("--show-toplevel")
        return git_root
    
        # * saves the current state to file
    def save_nca_state(self, _state, _name, _dir):
        
        # * set name
        if _name != None:
            self.name = _name
            
        if _dir != None:
            path = f'{self.repo_root}/obj/{_dir}/{self.name}'
        else:
            path = f'{self.repo_root}/obj/{self.name}'
            
        np_state = np.array(_state)
        np.save(path, np_state)
        
    # * saves the current self.voxels to a new .vox file
    def save_to_obj(self, _name=None, _dir=None, _voxel_size=0.1):
        
        # * return id no voxels found
        if self.voxels is None:
            print ('No voxels found in Vox object!')
            return
        
        # * set name
        if _name != None:
            self.name = _name
            
        if _dir != None:
            path = f'{self.repo_root}/obj/{_dir}/{self.name}'
        else:
            path = f'{self.repo_root}/obj/{self.name}'
        
        # * filter out voxels that are alive and visible
        filt_voxels = []
        size = self.rgba.shape[0]
        for x in range(size):
            for y in range(size):
                for z in range(size):
                    
                    # * make sure voxel is alive (alpha > 0.1)
                    if self.rgba[x, y, z, 3] > 0.1:
                        filt_voxels.append({'pos': [x, z, y], 'color': self.rgba[x, y, z]})
                        # # * make sure not on edge voxel
                        # if x != 0 and x != size-1 and y != 0 and y != size-1 and z != 0 and z != size-1:
                            
                        #     # * check if voxel is completly obstructed
                        #     if not (self.rgba[x+1, y, z, 3] > 0.1 and
                        #         self.rgba[x+1, y, z, 3] > 0.1 and
                        #         self.rgba[x-1, y, z, 3] > 0.1 and
                        #         self.rgba[x, y+1, z, 3] > 0.1 and
                        #         self.rgba[x, y-1, z, 3] > 0.1 and
                        #         self.rgba[x, y, z+1, 3] > 0.1 and
                        #         self.rgba[x, y, z-1, 3] > 0.1):
                        #             filt_voxels.append({'pos': [x, z, y], 'color': self.rgba[x, y, z]})
                        
                        # else:
                            
                            
        # * create .obj and .mtl files from filtered voxels
        mtl_file = open(f'{path}.mtl', 'w')
        obj_file = open(f'{path}.obj', 'w')
        obj_file.write(f'o {self.name}\n')
        obj_file.write('\n')
        obj_file.write(f'mtllib {self.name}.mtl\n')
        obj_file.write('\n')
        
        verts = []
        faces = []
        f = 0
        
        for i, voxel in enumerate(filt_voxels):
            pos = np.array(voxel['pos'])
            color = np.array(voxel['color'])
            
            a = color[3]
            if a < 0.95:
                a *= 0.5

            mtl_file.write(f'newmtl color{i}\n')
            mtl_file.write('Ka 0.0 0.0 0.0\n')
            mtl_file.write(f'Kd {color[0]} {color[1]} {color[2]}\n')
            mtl_file.write('Ks 0.0 0.0 0.0\n')
            mtl_file.write('Ns 0.0\n')
            mtl_file.write(f'd {a}\n')
            mtl_file.write('illum 0\n')
            mtl_file.write('\n')
            
            o = pos * _voxel_size
            s = _voxel_size
            verts.append(f'v {o[0]} {o[1]} {o[2]}\n')
            verts.append(f'v {o[0]} {o[1]} {o[2] + s}\n')
            verts.append(f'v {o[0] + s} {o[1]} {o[2] + s}\n')
            verts.append(f'v {o[0] + s} {o[1]} {o[2]}\n')
            verts.append(f'v {o[0]} {o[1] + s} {o[2]}\n')
            verts.append(f'v {o[0]} {o[1] + s} {o[2] + s}\n')
            verts.append(f'v {o[0] + s} {o[1] + s} {o[2] + s}\n')
            verts.append(f'v {o[0] + s} {o[1] + s} {o[2]}\n')

            faces.append(f'usemtl color{i}\n')
            faces.append(f'f {f+1} {f+2} {f+3}\n')
            faces.append(f'f {f+3} {f+4} {f+1}\n')
            faces.append(f'f {f+5} {f+6} {f+7}\n')
            faces.append(f'f {f+7} {f+8} {f+5}\n')
            faces.append(f'f {f+1} {f+2} {f+6}\n')
            faces.append(f'f {f+6} {f+5} {f+1}\n')
            faces.append(f'f {f+4} {f+3} {f+7}\n')
            faces.append(f'f {f+7} {f+8} {f+4}\n')
            faces.append(f'f {f+1} {f+5} {f+8}\n')
            faces.append(f'f {f+8} {f+4} {f+1}\n')
            faces.append(f'f {f+2} {f+6} {f+7}\n')
            faces.append(f'f {f+7} {f+3} {f+2}\n')
            faces.append('\n')
            
            f += 8
            
        for v in verts:
            obj_file.write(v)
        obj_file.write('\n')
        
        for f in faces:
            obj_file.write(f)
            
        mtl_file.close()
        obj_file.close()
    
    def load_from_tensor(self, _tensor, _name=None):
        if _name != None: self.name = _name
        _tensor = _tensor.cpu().detach()
        if len(_tensor.shape) == 5:
            _tensor = _tensor[0, ...]
        self.load_from_array(np.array(_tensor))
        return self
    
    def load_from_array(self, _array, _name=None):
        if _name != None: self.name = _name
        _array = _array.transpose(1, 2, 3, 0)
        rgba = _array
        rgba = np.clip(rgba, 0.0, 1.0)
        self.rgba = rgba[:, :, :, :4]
        self.rgb = self.rgba[:, :, :, :3]
        self.voxels = self.rgba[:, :, :, 3] > 0.1
        self.create_hex()
        return self
            
    def load_from_file(self, _filename):
        prts = _filename.split('/')
        self.name = prts[len(prts)-1].replace('.vox', '')
        self.vox_obj = parser.voxparser(_filename).parse()
        
        # * extract RGB data
        rgba = self.vox_obj.to_dense_rgba()
        rgba = np.transpose(rgba, (2, 0, 1, 3))
        rgba = np.flip(rgba, axis=2)
        rgba = rgba/255.0
        rgba = np.clip(rgba, 0.0, 1.0)
        self.rgba = rgba[:, :, :, :4]
        self.rgb = self.rgba[:, :, :, :3]
        self.voxels = self.rgba[:, :, :, 3] > 0.1
        self.create_hex()
        return self
    
    def shape(self):
        return self.voxels.shape
    
    def tensor(self):
        return torch.tensor(self.rgba, dtype=torch.float32).permute(3, 0, 1, 2)[None, ...]
    
    def numpy(self):
        return np.array(self.rgba).transpose(3, 0, 1, 2)[None, ...]
    
    def expand_coordinates(self, _indices):
        x, y, z = _indices
        x[1::2, :, :] += 1
        y[:, 1::2, :] += 1
        z[:, :, 1::2] += 1
        return x, y, z
    
    def explode(self, _data):
        shape_arr = np.array(_data.shape)
        size = shape_arr[:3]*2 - 1
        exploded = np.zeros(np.concatenate([size, shape_arr[3:]]), dtype=_data.dtype)
        exploded[::2, ::2, ::2] = _data
        return exploded
    
    def create_hex(self):
        self.hex = np.zeros(self.rgba[:, :, :, 0].shape, dtype='U9')
        size = self.hex.shape
        for x in range(size[0]):
            for y in range(size[1]):
                for z in range(size[2]):
                    self.hex[x, y, z] = to_hex(self.rgba[x, y, z], keep_alpha=True)
                    
        self.hex = self.explode(self.hex)
        self.voxels = self.explode(self.voxels)
        self.x, self.y, self.z = self.expand_coordinates(np.indices(np.array(self.voxels.shape)+1))
    
    def render(self, _pitch=10, _yaw=285, _show_grid=False, _print=True):
        # * render using plt
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        ax.voxels(self.x, self.y, self.z, self.voxels, facecolors=self.hex)
        ax.view_init(elev=_pitch, azim=_yaw)
        ax.set(xlabel='x', ylabel='y', zlabel='z')
        if not _show_grid: plt.axis('off')
        
        if _print: plt.show()
        else: plt.close()

        # * convert figure to numpy array RGB
        fig.canvas.draw()
        data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        return data
        
    def orbit(self, _filename=None, _turn=360, _delta=10, _zoom=1, _show_grid=False, _print=True):
        if _filename == None:
            _filename = self.name+'.mp4'
        with VideoWriter(filename=_filename) as vid:
            for i in range(0,_turn,_delta):
                img = self.render(_yaw=i, _show_grid=_show_grid, _print=False)
                vid.add(zoom(img, _zoom))
            if _print: vid.show()