import glm
import math
import numpy as np
import moderngl as mgl

PI = math.pi

class Plane:
    def __init__(self, _app):
        self.app = _app
        self.ctx = _app.ctx
        self.setup_plane()
        
    def setup_plane(self):
        # * setup cube 
        self.program = self.get_shader_program('plane')
        self.vao = self.get_vao()
        # * set uniforms
        self.program['u_proj'].write(self.app.player.m_proj)
        self.program['u_model'].write(glm.mat4())
        
    def update(self):
        self.program['u_view'].write(self.app.player.m_view)
        p = np.array(self.app.plane_pos)
        pos = glm.vec3(p[0], p[1], p[2])
        self.program['u_plane_pos'].write(pos)
        r = np.array(self.app.plane_rot)
        rot = glm.mat4(glm.quat(glm.vec3(r[0]*PI, r[1]*PI, r[2]*PI)))
        self.program['u_plane_rot'].write(rot)
        
    def render(self):
        self.vao.render(mgl.LINES)
        
    def destroy(self):
        self.vertex_vbo.release()
        self.color_vbo.release()
        self.program.release()
        self.vao.release()
        
    def get_vertex_data(self):
        v = 1
        vertices = [(-v, -v, 0), (v, -v, 0), (v, v, 0), (-v, v, 0)]
        lines = [(0, 1), (1, 2), (2, 3), (3, 0)]
        data = [vertices[ind] for line in lines for ind in line]
        data = np.array(data, dtype='f4')
        self.vertex_vbo = self.ctx.buffer(data)
        return self.vertex_vbo
    
    def get_color_data(self):
        colors = [(0.7, 0.7, 0.7),
                  (0.7, 0.7, 0.7),
                  (0.7, 0.7, 0.7), 
                  (0.7, 0.7, 0.7)]
        colors = [[val, val] for val in colors]
        data = np.array(colors, dtype='f4')
        self.color_vbo = self.ctx.buffer(data)
        return self.color_vbo
    
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