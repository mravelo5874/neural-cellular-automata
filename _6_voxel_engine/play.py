import os
import sys
import glm
import math
import threading
import moderngl as mgl
import pygame as pg
import pygame_gui as gui

from cube import Cube
from player import Player
from nca_simulator import NCASimulator

cwd = os.getcwd().split('\\')[:-1]
cwd = '/'.join(cwd)

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
        # * gui
        self.GUI = gui.UIManager(_win_size)
        self.SURF = pg.Surface(_win_size)
        hello_button = gui.elements.UIButton(relative_rect=pg.Rect((350, 275), (100, 50)),
                                             text='Say Hello',
                                             manager=self.GUI)
        # -------------------------- #
        
        # * get list of models
        self.models = next(os.walk(f'{cwd}/models/'))[1]
        self.curr_model = 5
        print (f'models: {self.models}')
        
        # * set opengl attributes
        pg.display.gl_set_attribute(pg.GL_CONTEXT_MAJOR_VERSION, 3)
        pg.display.gl_set_attribute(pg.GL_CONTEXT_MINOR_VERSION, 3)
        pg.display.gl_set_attribute(pg.GL_CONTEXT_PROFILE_MASK, pg.GL_CONTEXT_PROFILE_CORE)
        
        # * create opengl context
        self.WINDOW = pg.display.set_mode(self.WIN_SIZE, flags=pg.OPENGL|pg.DOUBLEBUF)
        
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
        
        # * init simulator
        self.sim = None
        def init_sim(_app):
            _app.sim = NCASimulator(_app.models[_app.curr_model])
        threading.Thread(target=init_sim, args=[self]).start()
        
        # * hide mouse cursor
        pg.event.set_grab(True)
        pg.mouse.set_visible(False)

        # * game is running
        self.is_running = True
        
    def update(self):
        # * calculate fps
        self.delta_time = self.clock.tick()
        self.time = pg.time.get_ticks() * 0.001
        pg.display.set_caption(f'fps: {self.clock.get_fps() :.0f}')
        
        # * update gui
        self.GUI.update(self.delta_time)
        
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
        
        # * TODO render gui using mgl
        # * links: https://stackoverflow.com/questions/76697818/gui-with-pygame-and-moderngl
        # *        https://stackoverflow.com/questions/66552579/how-can-i-draw-using-pygame-while-also-drawing-with-pyopengl/66552664#66552664
        self.WINDOW.blit(self.SURF, (0, 0))
        self.GUI.draw_ui(self.WINDOW)
        
        # * swap buffers
        pg.display.flip()
        
    def handle_events(self):
        for event in pg.event.get():
            # * quit application
            if event.type == pg.QUIT:
                self.is_running = False
                
            # * gui
            self.GUI.process_events(event)
            
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
                        
                # * step forward model (if paused)'
                if event.key == pg.K_RETURN:
                    if self.sim != None:
                        self.sim.step_forward()
                        
                # * load next model
                if event.key == pg.K_n:
                    if self.sim != None:
                        self.curr_model += 1
                        if self.curr_model > len(self.models)-1:
                            self.curr_model = 0
                        if self.sim.unload():
                            self.cube.destroy()
                            self.cube = Cube(self)
                            def init_sim(_app):
                                _app.sim = NCASimulator(_app.models[_app.curr_model])
                            threading.Thread(target=init_sim, args=[self]).start()
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