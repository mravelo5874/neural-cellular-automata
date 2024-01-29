from utility import *

class Position:
    def __init__(self, _pos):
        self.x = _pos[0]
        self.y = _pos[1]
        
    def xy(self):
        return np.array([self.x, self.y])
    
    def mod(self, _delta):
        self.x += _delta[0]
        self.y += _delta[1]
        
class Color:
    def __init__(self, _color):
        self.r = _color[0]
        self.g = _color[1]
        self.b = _color[2]
        self.a = _color[3]
        
    def rgb(self):
        return np.array([self.r, self.g, self.b])

    def rgba(self):
        return np.array([self.r, self.g, self.b, self.a])
    
    def mod(self, _delta):
        self.r += _delta[0]
        self.b += _delta[1]
        self.b += _delta[2]
        self.a += _delta[3]

class RadialCell:
    def __init__(self, _pos, _color, _angle, _hidden, _id):
        self.pos = Position(_pos)
        self.color = Color(_color)
        self.angle = _angle
        self.hidden = _hidden
        self.id = _id
        
    def state(self):
        return np.concatenate([self.color.rgba(), self.hidden])
    
    def update(self, _color_mod, _hidden_mod, _move_mod, _angle_mod):
        self.color.mod(_color_mod)
        self.pos.mod(_move_mod)
        
        self.hidden += _hidden_mod
        self.angle += _angle_mod
        self.angle % (np.pi*2.0)