import numpy as np
import matplotlib.pyplot as plt

import voxparser
from Video import VideoWriter, zoom, tile2d

class Vox(object):
    def __init__(self, _filename):
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
    
    def shape(self):
        return self.voxels.shape
    
    def render(self, _pitch=10, _yaw=280, _show=True):
        # * render using plt
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        ax.voxels(self.voxels, facecolors=self.rgb, edgecolors=self.rgb)
        ax.view_init(elev=_pitch, azim=_yaw)
        plt.axis('off')
        
        if _show: plt.show()
        else: plt.close()

        # * convert figure to numpy array RGB
        fig.canvas.draw()
        data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        return data
        
    def orbit(self, _filename=None, _turn=360, _delta=10, _zoom=1, _show=True):
        if _filename == None:
            _filename = self.name+'.mp4'
        with VideoWriter(filename=_filename) as vid:
            for i in range(0,_turn,_delta):
                img = self.render(_yaw=i, _show=False)
                vid.add(zoom(img, _zoom))
            if _show: vid.show()