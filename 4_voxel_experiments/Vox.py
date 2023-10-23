import torch
import numpy as np
import matplotlib.pyplot as plt
import voxparser
from Video import VideoWriter, zoom

class Vox(object):
    def __init__(self):
        self.name = 'unnamed_vox'
    
    def load_from_tensor(self, _tensor, _name=None):
        if _name != None: self.name = _name
        _tensor = _tensor.cpu()
        if len(_tensor.shape) == 5:
            _tensor = _tensor[0, ...]
        _tensor = torch.clamp(_tensor, min=0.0, max=1.0)
        self.load_from_array(np.array(_tensor))
        return self
    
    def load_from_array(self, _array, _name=None):
        if _name != None: self.name = _name
        _array = _array.transpose(1, 2, 3, 0)
        self.rgba = _array
        self.rgb = self.rgba[:, : , :, 0:3]
        self.voxels = self.rgba[:, :, :, 3] > 0.0
        return self
            
    def load_from_file(self, _filename):
        prts = _filename.split('/')
        self.name = prts[len(prts)-1].replace('.vox', '')
        self.vox_obj = voxparser.voxparser(_filename).parse()
        
        # * extract RGB data
        rgba = self.vox_obj.to_dense_rgba()
        rgba = np.transpose(rgba, (2, 0, 1, 3))
        rgba = np.flip(rgba, axis=2)
        self.rgba = rgba/255.0
        self.rgb = self.rgba[:, : , :, 0:3]
        
        # * create binary voxel
        self.voxels = self.rgba[:, :, :, 3] > 0.0
        return self
    
    def shape(self):
        return self.voxels.shape
    
    def tensor(self):
        return torch.tensor(self.rgba, dtype=torch.float32).permute(3, 0, 1, 2)[None, ...]
    
    def render(self, _pitch=10, _yaw=280, _show_grid=False, _print=True):
        # * render using plt
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        ax.voxels(self.voxels, facecolors=self.rgb, edgecolors=self.rgb)
        ax.view_init(elev=_pitch, azim=_yaw)
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