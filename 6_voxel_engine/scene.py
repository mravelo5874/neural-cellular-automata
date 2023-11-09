from settings import *
from _objects.chunk import Chunk

class Scene:
    def __init__(self, _app):
        self.app = _app
        self.chunk = Chunk(self.app)
        
    def update(self):
        pass
    
    def render(self):
        self.chunk.render()