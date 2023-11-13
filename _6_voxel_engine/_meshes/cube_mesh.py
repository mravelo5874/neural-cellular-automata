from _meshes.base_mesh import BaseMesh

class CubeMesh(BaseMesh):
    def __init__(self, _app):
        super().__init__()
        self.app = _app
        self.ctx = _app.ctx
        self.program = _app.shader_program.chunk
        
        self.vbo_format = '3u1 1u1 1u1'
        self.format_size = sum(int(fmt[:1]) for fmt in self.vbo_format.split())
        self.attrs = ('in_pos', 'voxel_id', 'face_id')
        self.vao = self.get_vao()
        
    def get_vertex_data(self):
        mesh = build_voxvol_mesh(
            _chunk_voxels=self.chunk.voxels,
            _format_size=self.format_size,
        )
        return mesh