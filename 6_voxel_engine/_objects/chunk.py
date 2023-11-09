from settings import *
from _meshes.chunk_mesh import ChunkMesh

class Chunk:
    def __init__(self, _app):
        self.app = _app
        self.voxels: np.array = self.build_voxels()
        self.mesh: ChunkMesh = None
        self.build_mesh()
        
    def build_mesh(self):
        self.mesh = ChunkMesh(self)
        
    def render(self):
        self.mesh.render()
    
    def build_voxels(self):
        # * create empty chunk
        voxels = np.zeros(CHUNK_VOL, dtype='uint8')
        
        # * fill chunk
        for x in range(CHUNK_SIZE):
            for y in range(CHUNK_SIZE):
                for z in range(CHUNK_SIZE):
                    voxels[x+CHUNK_SIZE*z+CHUNK_AREA*y] = x+y+z
        return voxels