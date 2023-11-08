from settings import *
import moderngl as mgl
import pygame as pg
import sys

class VoxelEngine:
    def __init__(self):
        # * setup pygame
        pg.init()
        pg.display.gl_get_attribute(pg.GL_CONTEXT_MAJOR_VERSION, 3)
        pg.display.gl_get_attribute(pg.GL_CONTEXT_MINOR_VERSION, 3)
        pg.display.gl_get_attribute(pg.GL_CONTEXT_PROFILE_MASK, pg.GL_CONTEXT_PROFILE_CORE)
        pg.display.gl_get_attribute(pg.GL_DEPTH_SIZE, 24)
        
        # * create context
        pg.display.set_mode(WIN_RES, flags=pg.OPENGL|pg.DOUBLEBUF)
        self.ctx = mgl.create_context()
        
        self.ctx.enable(flags=mgl.DEPTH_TEST|mgl.CULL_FACE|mgl.BLEND)
        self.ctx.gc_mode = 'auto'
    
    def update(self):
        pass
    
    def render(self):
        pass
    
    def handle_events(self):
        pass
    
    def run(self):
        pass
    
if __name__ == '__main__':
    app = VoxelEngine()
    app.run()