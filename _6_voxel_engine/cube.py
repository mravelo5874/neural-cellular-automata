import glm
import math
import numpy as np
import moderngl as mgl

PI = math.pi

class Cube:
    def __init__(self, _app):
        self.app = _app
        self.ctx = _app.ctx
        self.setup_cube()
        
    def setup_cube(self):
        # * setup cube 
        self.vbo = self.get_vbo()
        self.program = self.get_shader_program('cube')
        self.vao = self.get_vao()
        # * set texture3d
        self.texture = None
        self.blend = False
        self.program['u_volume'] = 0
        # * set uniforms
        self.program['u_proj'].write(self.app.player.m_proj)
        self.program['u_model'].write(glm.mat4())
        self.program['u_eye'].write(self.app.player.pos)
        
    def update(self):
        self.program['u_view'].write(self.app.player.m_view)
        self.program['u_eye'].write(self.app.player.pos)
        
        # * set plane boolean
        self.program['u_plane_on'].write(glm.bool_(self.app.SHOW_PLANE))
        
        # * set plane position 
        p = np.array(self.app.plane_pos)
        pos = glm.vec3(p[0], p[1], p[2])
        self.program['u_plane_pos'].write(pos)
        
        # * set plane normal vector
        r = np.array(self.app.plane_rot)
        rot = glm.mat4(glm.quat(glm.vec3(r[0]*PI, r[1]*PI, r[2]*PI)))
        norm = glm.vec4(0.0, 0.0, 1.0, 1.0)
        norm = glm.normalize(rot * norm)
        self.program['u_plane_norm'].write(norm.xyz)
        
        # * update volume if sim exists and is loaded
        if self.app.sim != None:
            if self.app.sim.is_loaded:            
                # * setup texture3d
                if self.texture == None:
                    self.init_texture()
                    self.app.sim.run()
                    
                # * update volume data
                data = self.app.sim.get_data()
                self.texture.write(data)
            
    def init_texture(self):
        s = self.app.sim.size
        # * 8 bits per component, 4 components per voxel
        self.texture = self.ctx.texture3d([s, s, s], 4, dtype='f1')
        self.texture.swizzle = 'RGBA'
        self.texture.filter = mgl.NEAREST, mgl.NEAREST
        self.texture.repeat_x = False
        self.texture.repeat_y = False
        self.texture.repeat_z = False
        data = self.app.sim.get_data() # self.dummy_data(s) # 
        self.texture.write(data)
        self.texture.use()
        
    def dummy_data(self, _size):
        data = np.zeros((_size, _size, _size, 4))
        for x in range(_size):
            for y in range(_size):
                for z in range(_size):
                    data[x, y, z,] = [x/_size, y/_size, z/_size, 1.0]
        data = data.reshape((4*_size*_size*_size))*255
        data = data.astype(np.uint8)
        return data.tobytes()
        
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
        vertices = [(-1, -1, 1), (1, -1, 1), (1, 1, 1), (-1, 1, 1),
                    (-1, 1, -1), (-1, -1, -1), (1, -1, -1), (1, 1, -1)]
        indicies = [(0, 2, 3), (0, 1, 2),
                    (1, 7, 2), (1, 6, 7),
                    (6, 5, 4), (4, 7, 6),
                    (3, 4, 5), (3, 5, 0),
                    (3, 7, 4), (3, 2, 7),
                    (0, 6, 1), (0, 5, 6)]
        vertex_data = self.get_data(vertices, indicies)
        # print (f'vertex data: {vertex_data}')
        return vertex_data
    
    @staticmethod
    def get_data(_vertices, _indices):
        data = [_vertices[ind] for triangle in _indices for ind in triangle]
        return np.array(data, dtype='f4')
    
    def get_vbo(self):
        vertex_data = self.get_vertex_data()
        vbo = self.ctx.buffer(vertex_data)
        return vbo
    
    def set_blend(self, _opt):
        if self.texture != None:
            self.blend = _opt
            if self.blend:
                self.texture.filter = mgl.LINEAR, mgl.LINEAR
            else:
                self.texture.filter = mgl.NEAREST, mgl.NEAREST
            return self.blend
    
    def toggle_blend(self):
        if self.texture != None:
            self.blend = not self.blend
            return self.set_blend(self.blend)
    
    def get_shader_program(self, _name):
        with open(f'shaders/{_name}.vert') as file:
            vert = file.read()
        with open(f'shaders/{_name}.frag') as file:
            frag = file.read()
         
        program = self.ctx.program(vertex_shader=vert, fragment_shader=frag)
        return program