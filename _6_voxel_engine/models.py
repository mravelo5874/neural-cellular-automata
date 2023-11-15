import glm
import numpy as np

class Cube:
    def __init__(self, _app):
        self.app = _app
        self.ctx = _app.ctx
        self.vbo = self.get_vbo()
        self.program = self.get_shader_program('default')
        self.vao = self.get_vao()
        self.on_init()
        
    def on_init(self):
        self.program['m_proj'].write(self.app.player.m_proj)
        self.program['m_model'].write(glm.mat4())

    def update(self):
        self.program['m_view'].write(self.app.player.m_view)
        
    def render(self):
        self.vao.render()
        
    def destroy(self):
        self.vbo.release()
        self.program.release()
        self.vao.release()
        
    def get_vao(self):
        vao = self.ctx.vertex_array(self.program, [(self.vbo, '3f', 'in_pos')])
        return vao
        
    def get_vertex_data(self):
        vertices = [(-1, -1, +1), (+1, -1, +1), (+1, +1, +1), (-1, +1, +1),
                    (-1, +1, -1), (-1, -1, -1), (+1, -1, -1), (+1, +1, -1)]
        indicies = [(0, 2, 3), (0, 1, 2),
                    (1, 7, 2), (1, 6, 7),
                    (6, 5, 4), (4, 7, 6),
                    (3, 4, 5), (3, 5, 0),
                    (3, 7, 4), (3, 2, 7),
                    (0, 6, 1), (0, 5, 6)]
        vertex_data = self.get_data(vertices, indicies)
        return vertex_data
    
    @staticmethod
    def get_data(_vertices, _indices):
        data = [_vertices[ind] for triangle in _indices for ind in triangle]
        return np.array(data, dtype='f4')
    
    def get_vbo(self):
        vertex_data = self.get_vertex_data()
        vbo = self.ctx.buffer(vertex_data)
        return vbo
    
    def get_shader_program(self, _name):
        with open(f'shaders/{_name}.vert') as file:
            vert = file.read()
        with open(f'shaders/{_name}.frag') as file:
            frag = file.read()   
         
        program = self.ctx.program(vertex_shader=vert, fragment_shader=frag)
        return program      