import glm
import ctypes
import numpy as np
import moderngl as mgl

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
        
    def get_vao(self):
        vao = self.ctx.vertex_array(self.program, [(self.vbo, '2f4 2f4', 'in_pos', 'in_uv')])
        return vao
    
    def get_vbo(self):
        vbo = self.ctx.buffer(None, reserve=6*4*4)
        return vbo
    
    def update(self):
        pass
    
    def convert_vertex(pt, surface):
        return pt[0] / surface.get_width() * 2 - 1, 1 - pt[1] / surface.get_height() * 2 
    
    def render(self, _surf, _sprite):
        corners = [
            mgl.convert_vertex(_sprite.rect.bottomleft, _surf),
            mgl.convert_vertex(_sprite.rect.bottomright, _surf),
            mgl.convert_vertex(_sprite.rect.topright, _surf),
            mgl.convert_vertex(_sprite.rect.topleft, _surf)] 
        vertices_quad_2d = (ctypes.c_float * (6*4))(
            *corners[0], 0.0, 1.0, 
            *corners[1], 1.0, 1.0, 
            *corners[2], 1.0, 0.0,
            *corners[0], 0.0, 1.0, 
            *corners[2], 1.0, 0.0, 
            *corners[3], 0.0, 0.0)
        
        
    
    def destroy(self):
        pass
    
    def get_shader_program(self, _name):
        with open(f'shaders/{_name}.vert') as file:
            vert = file.read()
        with open(f'shaders/{_name}.frag') as file:
            frag = file.read()
         
        program = self.ctx.program(vertex_shader=vert, fragment_shader=frag)
        return program