from settings import *
from _6_voxel_engine._meshes.voxvol_mesh import VoxVolMesh

class Chunk:
    def __init__(self, _app):
        self.app = _app
        self.voxels: np.array = self.build_voxels()
        self.mesh: VoxVolMesh = None
        self.build_mesh()
        
    def build_mesh(self):
        self.mesh = VoxVolMesh(self)
        
    def render(self):
        self.mesh.render()
    
    def build_voxels(self):
        # * create empty chunk
        voxels = np.zeros(CHUNK_VOL, dtype='uint8')
        
        # * fill chunk
        for x in range(CHUNK_SIZE):
            for y in range(CHUNK_SIZE):
                for z in range(CHUNK_SIZE):
                    voxels[x+CHUNK_SIZE*z+CHUNK_AREA*y] = (
                        x+y+z if int(glm.simplex(glm.vec3(x, y, z)*0.1)+1) else 0
                    )
        return voxels