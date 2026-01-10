import numpy as np
import moderngl as mgl

class Crosshair:
    def __init__(self, _app):
        self.app = _app
        self.ctx = _app.ctx
        self.setup_crosshair()
        
    def setup_crosshair(self):
        # * setup cube 
        self.program = self.get_shader_program('crosshair')
        self.vao = self.get_vao()
        
    def render(self):
        self.vao.render(mgl.POINTS)
        
    def get_vao(self):
        # Coordinates for the center of the screen
        x = 0
        y = 0
        coords = np.array([x, y], dtype='f4')

        # Create a buffer for the coordinates
        vbo = self.ctx.buffer(coords)
        vao = self.ctx.vertex_array(self.program, [
            (vbo, '2f', 'in_vert'),
        ])
        return vao
        
    def get_shader_program(self, _name):
        with open(f'shaders/{_name}.vert') as file:
            vert = file.read()
        with open(f'shaders/{_name}.frag') as file:
            frag = file.read()
         
        program = self.ctx.program(vertex_shader=vert, fragment_shader=frag)
        return program