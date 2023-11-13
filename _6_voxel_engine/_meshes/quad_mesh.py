import numpy as np
from settings import *
from _meshes.base_mesh import BaseMesh

class QuadMesh(BaseMesh):
    def __init__(self, _app):
        super().__init__()
        
        self.app = _app
        self.ctx = _app.ctx
        self.program = _app.shader_program.quad
        
        self.vbo_format = '3f 3f'
        self.attrs = ('in_pos', 'in_color')
        self.vao = self.get_vao()
        
    def get_vertex_data(self):
        vertices = [
            (0.5, 0.5, 0.0), (-0.5, 0.5, 0.0), (-0.5, -0.5, 0.0), 
            (0.5, 0.5, 0.0), (-0.5, -0.5, 0.0), (0.5, -0.5, 0.0)
        ]
        colors = [
            (0, 1, 0), (1, 0, 0), (1, 1, 0),
            (0, 1, 0), (1, 1, 0), (0, 0, 1)    
        ]
        vertex_data = np.hstack([vertices, colors], dtype='float32')
        return vertex_data