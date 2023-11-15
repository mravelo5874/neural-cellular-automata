import moderngl as mgl
import pygame as pg
import sys
import glm
import math
import threading

from cube import Cube
from player import Player
from nca_simulator import NCASimulator

class VoxelEngine:
    def __init__(self, _win_size=(1200, 800)):
        # * init pygame modules
        pg.init()
        
        # -------- settings -------- #
        # * window
        self.WIN_SIZE = _win_size
        self.BG_COLOR = (1.0, 1.0, 1.0)
        # * camera
        self.ASPECT_RATIO = _win_size[0]/_win_size[1]
        self.FOV_DEG = 50
        self.V_FOV = glm.radians(self.FOV_DEG)
        self.H_FOV = 2*math.atan(math.tan(self.V_FOV*0.5)*self.ASPECT_RATIO)
        self.NEAR = 0.1
        self.FAR = 2000
        self.MAX_PITCH = glm.radians(89)
        # * player
        self.PLAYER_SPEED = 0.002
        self.PLAYER_ROT_SPEED = 0.005
        self.PLAYER_POS = glm.vec3(1, 0, 4)
        self.MOUSE_SENS = 0.002
        # * model
        self.model_name = 'cowboy16_yawiso8'
        # -------------------------- #
        
        # * set opengl attributes
        pg.display.gl_set_attribute(pg.GL_CONTEXT_MAJOR_VERSION, 3)
        pg.display.gl_set_attribute(pg.GL_CONTEXT_MINOR_VERSION, 3)
        pg.display.gl_set_attribute(pg.GL_CONTEXT_PROFILE_MASK, pg.GL_CONTEXT_PROFILE_CORE)
        
        # * create opengl context
        pg.display.set_mode(self.WIN_SIZE, flags=pg.OPENGL|pg.DOUBLEBUF)
        
        # * hide mouse cursor
        pg.event.set_grab(True)
        pg.mouse.set_visible(False)
        
        # * use opengl context
        self.ctx = mgl.create_context()
        self.ctx.front_face = 'cw'
        self.ctx.enable(flags=mgl.DEPTH_TEST|mgl.CULL_FACE)
        self.ctx.disable(flags=mgl.BLEND)
        
        # * create clock to track time
        self.clock = pg.time.Clock()
        self.delta_time = 0
        self.time = 0
        
        # * init player and cube
        self.player = Player(self)
        self.cube = Cube(self)
        
        # * init simulator
        self.sim = None
        def init_sim(_app):
            _app.sim = NCASimulator(_app.model_name)
        threading.Thread(target=init_sim, args=[self]).start()

        # * game is running
        self.is_running = True
        
    def update(self):
        # * calculate fps
        self.delta_time = self.clock.tick()
        self.time = pg.time.get_ticks() * 0.001
        pg.display.set_caption(f'fps: {self.clock.get_fps() :.0f}')
        
        # * update player and cube
        self.player.update()
        self.cube.update()
        
    def render(self):
        # * clear framebuffer
        self.ctx.clear(color=self.BG_COLOR)
        
        # * render cube
        self.cube.render()
        
        # * swap buffers
        pg.display.flip()
        
    def handle_events(self):
        for event in pg.event.get():
            if event.type == pg.QUIT or (event.type == pg.KEYDOWN and event.key == pg.K_ESCAPE):
                self.is_running = False
                
    def run(self):
        while self.is_running:
            self.handle_events()
            self.update()
            self.render()
            
        # * destroy scene
        self.cube.destroy()
        
        # * quit application
        pg.quit()
        sys.exit()

        
if __name__ == '__main__':
    app = VoxelEngine()
    app.run()