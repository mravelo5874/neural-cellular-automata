from _meshes.base_mesh import BaseMesh
from _meshes.chunk_mesh_builder import build_chunk_mesh

class ChunkMesh(BaseMesh):
    def __init__(self, _chunk):
        super().__init__()
        self.chunk = _chunk
        self.app = _chunk.app
        self.ctx = _chunk.app.ctx
        self.program = _chunk.app.shader_program.chunk
        
    def get_vertex_data(self):
        mesh = build_chunk_mesh(
            _chunk_voxels=self.chunk.voxels
        )
        return mesh