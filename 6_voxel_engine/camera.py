from settings import *

class Camera:
    def __init__(self, _pos, _yaw, _pitch):
        self.pos = glm.vec3(_pos)
        self.yaw = glm.radians(_yaw)
        self.pitch = glm.radians(_pitch)
        
        self.up = glm.vec3(0, 1, 0)
        self.right = glm.vec3(1, 0, 0)
        self.forward = glm.vec3(0, 0, -1)
        
        self.m_proj = glm.perspective(VERT_FOV, ASPECT_RATIO, NEAR, FAR)
        self.m_view = glm.mat4()
    
    def update(self):
        self.update_vectors()
        self.update_view_matrix()
    
    def update_view_matrix(self):
        self.m_view = glm.lookAt(self.pos, self.pos+self.forward, self.up)
        
    def update_vectors(self):
        # * update forward vector
        self.forward.x = glm.cos(self.yaw)*glm.cos(self.pitch)
        self.forward.y = glm.sin(self.pitch)
        self.forward.z = glm.sin(self.yaw)*glm.cos(self.pitch)
        self.forward = glm.normalize(self.forward)
        
        # * update right and up vectors
        self.right = glm.normalize(glm.cross(self.forward, glm.vec3(0, 1, 0)))
        self.up = glm.normalize(glm.cross(self.right, self.forward))

    def rotate_pitch(self, _delta_y):
        self.pitch -= _delta_y
        self.pitch = glm.clamp(self.pitch, -PITCH_MAX, PITCH_MAX)
        
    def rotate_yaw(self, _delta_x):
        self.yaw += _delta_x
        
    def move_left(self, _vel):
        self.pos -= self.right*_vel
        
    def move_right(self, _vel):
        self.pos += self.right*_vel
        
    def move_up(self, _vel):
        self.pos += self.up*_vel
        
    def move_down(self, _vel):
        self.pos -= self.up*_vel
        
    def move_forward(self, _vel):
        self.pos += self.forward*_vel
    
    def move_back(self, _vel):
        self.pos -= self.forward*_vel