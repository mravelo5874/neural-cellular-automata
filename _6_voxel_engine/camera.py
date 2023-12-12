import glm

FOV = 50 # * degrees
NEAR = 0.1
FAR = 100

class Camera:
    def __init__(self, _app, _pos, _yaw, _pitch):
        self.pos = _pos
        self.yaw = _yaw
        self.pitch = _pitch
        self.max_pitch = _app.MAX_PITCH
        
        self.up = glm.vec3(0, 0, 1)
        self.right = glm.vec3(0, 1, 0)
        self.forward = glm.vec3(0, 0, -1)
        
        self.m_proj = glm.perspective(_app.V_FOV, _app.ASPECT_RATIO, _app.NEAR, _app.FAR)
        self.m_view = glm.mat4()
        
    def update(self):
        self.update_vectors()
        self.update_view_matrix()
        
    def update_view_matrix(self):
        self.m_view = glm.lookAt(self.pos, self.pos + self.forward, self.up)
        
    def update_vectors(self):
        self.forward.x = glm.sin(self.yaw) * glm.cos(self.pitch)
        self.forward.y = glm.cos(self.yaw) * glm.cos(self.pitch)
        self.forward.z = glm.sin(self.pitch)
        
        self.forward = glm.normalize(self.forward)
        self.right = glm.normalize(glm.cross(self.forward, glm.vec3(0, 0, 1)))
        self.up = glm.normalize(glm.cross(self.right, self.forward))
        
    def rotate_pitch(self, _dy):
        self.pitch -= _dy
        self.pitch = glm.clamp(self.pitch, -self.max_pitch, self.max_pitch)
        
    def rotate_yaw(self, _dx):
        self.yaw += _dx
        
    def move_left(self, _v):
        self.pos -= self.right * _v
    
    def move_right(self, _v):
        self.pos += self.right * _v
        
    def move_down(self, _v):
        self.pos -= self.up * _v
    
    def move_up(self, _v):
        self.pos += self.up * _v
    
    def move_back(self, _v):
        self.pos -= self.forward * _v
        
    def move_forward(self, _v):
        self.pos += self.forward * _v