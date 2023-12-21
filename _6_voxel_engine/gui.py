import moderngl as mgl
import pygame as pg
import numpy as np

class Gui:
    def __init__(self, _app):
        self.app = _app
        self.ctx = _app.ctx
        self.setup_gui()
        
    def setup_gui(self):
        # * setup cube 
        self.vbo = self.get_vbo()
        self.program = self.get_shader_program('gui')
        self.vao = self.get_vao()
        # * set texture
        self.texture = self.ctx.texture(size=self.app.WIN_SIZE, components=4, dtype='f1')
        self.program['u_texture'] = 0
        self.texture.swizzle = 'RGBA'
        self.texture.filter = mgl.NEAREST, mgl.NEAREST
        self.texture.repeat_x = False
        self.texture.repeat_y = False
        self.texture.repeat_z = False
        data = pg.image.tostring(self.app.SURF, 'RGBA', True)
        self.texture.write(data)
        self.texture.use()
        
    def get_vbo(self):
        vertex_data = self.get_vertex_data()
        vbo = self.ctx.buffer(vertex_data)
        return vbo
            
    def get_vertex_data(self):
        vertices = [(-1, -1), (-1,  1), ( 1, -1),
                    ( 1, -1), (-1,  1), ( 1,  1)]
        uvs = [(0, 0), (0, 1), (1, 0),
               (1, 0), (0, 1), (1, 1)]
        vertex_data = []
        for i in range(len(vertices)):
            vertex_data.append([vertices[i], uvs[i]])
        vertex_data = np.array(vertex_data, dtype='f4')
        return vertex_data
        
    def get_vao(self):
        vao = self.ctx.vertex_array(self.program, [(self.vbo, '2f 2f', 'in_pos', 'in_uvs')])
        return vao
    
    def update(self, _surf):
        rgba = pg.image.tostring(_surf, 'RGBA', True)
        self.texture.write(rgba)
        
    def render(self):
        self.vao.render()
        
    def destroy(self):
        self.vbo.release()
        self.program.release()
        self.vao.release()
    
    def get_shader_program(self, _name):
        with open(f'shaders/{_name}.vert') as file:
            vert = file.read()
        with open(f'shaders/{_name}.frag') as file:
            frag = file.read()
         
        program = self.ctx.program(vertex_shader=vert, fragment_shader=frag)
        return program