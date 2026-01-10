import glm
import numpy as np
import moderngl as mgl

class WireFrame:
    def __init__(self, _app):
        self.app = _app
        self.ctx = _app.ctx
        self.setup_wireframe()
        
    def setup_wireframe(self):
        # * setup cube 
        self.program = self.get_shader_program('wireframe')
        self.vao = self.get_vao()
        # * set uniforms
        self.program['u_proj'].write(self.app.player.m_proj)
        self.program['u_model'].write(glm.mat4())
        
    def update(self):
        self.program['u_view'].write(self.app.player.m_view)
    
    def render(self):
        self.vao.render(mgl.LINES)
        
    def destroy(self):
        self.vertex_vbo.release()
        self.color_vbo.release()
        self.program.release()
        self.vao.release()
        
    def get_vertex_data(self):
        v = 1.01
        vertices = [(-v, -v, v), (v, -v, v), (v, v, v), (-v, v, v),
                    (-v, v, -v), (-v, -v, -v), (v, -v, -v), (v, v, -v)]
        lines = [(0, 1), (1, 2), (2, 3), (3, 0),
                 (4, 5), (5, 6), (6, 7), (7, 4),
                 (0, 5), (1, 6), (2, 7), (3, 4)]
        data = [vertices[ind] for line in lines for ind in line]
        data = np.array(data, dtype='f4')
        self.vertex_vbo = self.ctx.buffer(data)
        return self.vertex_vbo
    
    def get_color_data(self):
        colors = [(1.0, 0.0, 0.0),
                  (0.0, 1.0, 0.0),
                  (1.0, 0.0, 0.0), 
                  (0.0, 1.0, 0.0),
                
                  (0.0, 1.0, 0.0),
                  (1.0, 0.0, 0.0),
                  (0.0, 1.0, 0.0),
                  (1.0, 0.0, 0.0),
                
                  (0.0, 0.0, 1.0),
                  (0.0, 0.0, 1.0),
                  (0.0, 0.0, 1.0),
                  (0.0, 0.0, 1.0)]
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