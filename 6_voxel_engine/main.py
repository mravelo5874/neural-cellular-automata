import moderngl as mgl
import pygame as pg
import sys

from settings import *
from shader_program import ShaderProgram
from scene import Scene
from player import Player

class VoxelEngine:
    def __init__(self):
        # * setup pygame + gl attributes
        pg.init()
        pg.display.gl_set_attribute(pg.GL_CONTEXT_MAJOR_VERSION, 3)
        pg.display.gl_set_attribute(pg.GL_CONTEXT_MINOR_VERSION, 3)
        pg.display.gl_set_attribute(pg.GL_CONTEXT_PROFILE_MASK, pg.GL_CONTEXT_PROFILE_CORE)
        pg.display.gl_set_attribute(pg.GL_DEPTH_SIZE, 24)
        pg.display.set_mode(WIN_RES, flags=pg.OPENGL|pg.DOUBLEBUF)
        
        # * create context
        self.ctx = mgl.create_context()
        
        # * fragment depth testing, culling faces and color blending
        self.ctx.enable(flags=mgl.DEPTH_TEST|mgl.CULL_FACE)
        self.ctx.disable(flags=mgl.BLEND)
        
        # * automatic garbage collection
        self.ctx.gc_mode = 'auto'
        
        # setup clock + fps
        self.clock = pg.time.Clock()
        self.delta_time = 0
        self.time = 0
        
        # * lock mouse cursor
        pg.event.set_grab(True)
        pg.mouse.set_visible(False)

        self.is_running = True
        self.on_init()
        
    def on_init(self):
        self.player = Player(self)
        self.shader_program = ShaderProgram(self)
        self.scene = Scene(self)
    
    def update(self):
        self.player.update()
        self.shader_program.update()
        self.scene.update()
        
        # * update time variables
        self.delta_time = self.clock.tick()
        self.time = pg.time.get_ticks()*0.001
        
        # * show fps
        pg.display.set_caption(f'fps: {self.clock.get_fps():.0f}')
    
    def render(self):
        # * clear frame and depth buffers
        self.ctx.clear(color=BG_COLOR)
        
        # * render the scene
        self.scene.render()
        
        # * flip it!
        pg.display.flip()
    
    def handle_events(self):
        for event in pg.event.get():
            # * exit app
            if event.type == pg.QUIT or (event.type == pg.KEYDOWN and event.key == pg.K_ESCAPE):
                self.is_running = False
    
    def run(self):
        # * game loop
        while self.is_running:
            self.handle_events()
            self.update()
            self.render()
        pg.quit()
        sys.exit()
    
if __name__ == '__main__':
    app = VoxelEngine()
    app.run()