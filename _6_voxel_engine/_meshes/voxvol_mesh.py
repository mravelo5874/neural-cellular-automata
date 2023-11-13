from _meshes.base_mesh import BaseMesh
from _6_voxel_engine._meshes.voxvol_mesh_builder import build_chunk_mesh

class ChunkMesh(BaseMesh):
    def __init__(self, _chunk):
        super().__init__()
        self.chunk = _chunk
        self.app = _chunk.app
        self.ctx = _chunk.app.ctx
        self.program = _chunk.app.shader_program.chunk
        
        self.vbo_format = '3u1 1u1 1u1'
        self.format_size = sum(int(fmt[:1]) for fmt in self.vbo_format.split())
        self.attrs = ('in_pos', 'voxel_id', 'face_id')
        self.vao = self.get_vao()
        
    def get_vertex_data(self):
        mesh = build_chunk_mesh(
            _chunk_voxels=self.chunk.voxels,
            _format_size=self.format_size,
        )
        return mesh