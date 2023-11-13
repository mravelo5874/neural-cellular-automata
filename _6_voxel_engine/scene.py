from settings import *
from _6_voxel_engine._objects.voxvol import Chunk

from _5_voxel_nca.scripts.vox.Vox import Vox

class Scene:
    def __init__(self, _app):
        self.app = _app
        self.vox = Vox().load_from_file(_app._VOX_FILE_)
        self.chunk = Chunk(self.app)
        
    def update(self):
        pass
    
    def render(self):
        self.chunk.render()