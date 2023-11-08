from settings import *

class ShaderProgram:
    def __init__(self, _app):
        self.app = _app
        self.ctx = _app.ctx
        self.player = _app.player
        # -------- shaders -------- #
        self.quad = self.get_program(_shader_name='quad')
        # ------------------------- #
        
        self.set_uniforms_on_init()
        
    def set_uniforms_on_init(self):
        self.quad['m_proj'].write(self.player.m_proj)
        self.quad['m_model'].write(glm.mat4())
    
    def update(self):
        self.quad['m_view'].write(self.player.m_view)
    
    def get_program(self, _shader_name):
        # * get vertex and fragment shader programs
        with open(f'_shaders/{_shader_name}.vert') as file:
            vert = file.read()
        with open(f'_shaders/{_shader_name}.frag') as file:
            frag = file.read()
        
        # * create program
        program = self.ctx.program(vertex_shader=vert, fragment_shader=frag)
        return program