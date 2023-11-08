from settings import *
from _meshes.quad_mesh import QuadMesh

class Scene:
    def __init__(self, _app):
        self.app = _app
        self.quad = QuadMesh(_app)
        
    def update(self):
        pass
    
    def render(self):
        self.quad.render()