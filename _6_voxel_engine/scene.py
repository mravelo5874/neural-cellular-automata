from settings import *

from _objects.voxvol import VoxVol
from _objects.cube import Cube

from _5_voxel_nca.scripts.vox.Vox import Vox

class Scene:
    def __init__(self, _app):
        self.app = _app
        self.vox = Vox().load_from_file(_app._VOX_FILE_)
        self.cube = Cube()
        self.chunk = VoxVol(self.app)
        
    def update(self):
        pass
    
    def render(self):
        self.chunk.render()