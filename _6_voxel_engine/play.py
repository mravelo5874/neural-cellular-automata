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
        self.BG_COLOR = (178/255, 196/255, 209/255)
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
        self.PLAYER_POS = glm.vec3(-3, 0, 3)
        self.MOUSE_SENS = 0.002
        self.CREATIVE_MODE = True
        # * model
        self.model_name = 'oak_aniso'
        # -------------------------- #
        
        # * set opengl attributes
        pg.display.gl_set_attribute(pg.GL_CONTEXT_MAJOR_VERSION, 3)
        pg.display.gl_set_attribute(pg.GL_CONTEXT_MINOR_VERSION, 3)
        pg.display.gl_set_attribute(pg.GL_CONTEXT_PROFILE_MASK, pg.GL_CONTEXT_PROFILE_CORE)
        
        # * create opengl context
        pg.display.set_mode(self.WIN_SIZE, flags=pg.OPENGL|pg.DOUBLEBUF)
        
        # * use opengl context
        self.ctx = mgl.create_context()
        self.ctx.front_face = 'cw'
        self.ctx.enable(flags=mgl.DEPTH_TEST|mgl.CULL_FACE)
        
        # * create clock to track time
        self.clock = pg.time.Clock()
        self.delta_time = 0
        self.time = 0
        
        # * init player and cube
        self.player = Player(self)
        self.cube = Cube(self)
        
        # * hide mouse cursor
        pg.event.set_grab(True)
        pg.mouse.set_visible(False)
        
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
        
        # * update player
        if self.CREATIVE_MODE:
            self.player.update()
            
        # * update cube
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
            # * quit application
            if event.type == pg.QUIT:
                self.is_running = False
            
            # -------- key press events -------- #
            if event.type == pg.KEYDOWN:
            
                # * exit creative mode if press ESC key
                if event.key == pg.K_ESCAPE:
                    if self.CREATIVE_MODE:
                        self.CREATIVE_MODE = False
                        # * show mouse cursor
                        pg.event.set_grab(False)
                        pg.mouse.set_visible(True)
                        
                # * toggle voxel blend
                if event.key == pg.K_b:
                    self.cube.toggle_blend()
                    
                # * reset model
                if event.key == pg.K_r:
                    if self.sim != None:
                        self.sim.reset()
                        
                # * pause/unpause model
                if event.key == pg.K_p:
                    if self.sim != None:
                        self.sim.toggle_pause()
            # ---------------------------------- #
                    
                    
            # ---------- mouse events ---------- #
            if pg.mouse.get_pressed(3)[0]:
                # * enter creative mode if click on screen
                if not self.CREATIVE_MODE:
                    self.CREATIVE_MODE = True
                    # * hide mouse cursor
                    pg.event.set_grab(True)
                    pg.mouse.set_visible(False)
            # ---------------------------------- #
            
    def run(self):
        while self.is_running:
            self.handle_events()
            self.update()
            self.render()
            
        # * destroy scene
        self.cube.destroy()
        
        # * quit application
        print ('exiting application...')
        pg.quit()
        sys.exit()

        
if __name__ == '__main__':
    app = VoxelEngine()
    app.run()