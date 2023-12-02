import glm
import numpy as np
import moderngl as mgl
from utils import Utils as utils

class Vector:
    def __init__(self, _app):
        self.app = _app
        self.ctx = _app.ctx
        self.setup_vector()
        
    def setup_vector(self):
        # * setup cube 
        self.program = self.get_shader_program('vector')
        self.vao = self.get_vao()
        # * set uniforms
        self.program['u_proj'].write(self.app.player.m_proj)
        self.program['u_model'].write(glm.mat4())
        
    def update(self):
        self.program['u_view'].write(self.app.player.m_view)
        self.vao = self.get_vao()
    
    def render(self):
        self.vao.render(mgl.LINES)
        
    def destroy(self):
        self.program.release()
        self.vao.release()
        
    def get_vertex_data(self):
        pos, vec = self.app.my_vector
        if pos == None or vec == None:
            pos, vec = glm.vec3(-1), glm.vec3(-1)
        t_min, t_max = utils.ray_box_intersection(pos, vec, glm.vec3(-1), glm.vec3(1))
        end = pos+vec*t_min
        vertices = [pos, end]
        data = np.array(vertices, dtype='f4')
        vbo = self.ctx.buffer(data)
        return vbo
    
    def get_color_data(self):
        colors = [(1.0, 0.0, 1.0),
                  (0.0, 0.0, 0.0)]
        colors = [[val, val] for val in colors]
        data = np.array(colors, dtype='f4')
        vbo = self.ctx.buffer(data)
        return vbo
    
    def get_vao(self):
        pos_vbo = self.get_vertex_data()
        color_vbo = self.get_color_data()
        vao = self.ctx.vertex_array(self.program, [
            (pos_vbo, '3f', 'in_pos'),
            (color_vbo, '3f', 'in_color'),
        ])
        return vao
    
    def get_shader_program(self, _name):
        with open(f'shaders/{_name}.vert') as file:
            vert = file.read()
        with open(f'shaders/{_name}.frag') as file:
            frag = file.read()
         
        program = self.ctx.program(vertex_shader=vert, fragment_shader=frag)
        return program