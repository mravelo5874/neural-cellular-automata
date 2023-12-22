import os
import sys
import glm
import math
import time
import threading
import moderngl as mgl
import pygame as pg
import pygame_gui as gui
from pygame_gui.core import ObjectID as obj

from cube import Cube
from wireframe import WireFrame
from axis import Axis
from voxel import Voxel
from crosshair import Crosshair
from player import Player
from vector import Vector
from gui import Gui
from nca_simulator import NCASimulator

cwd = os.getcwd().split('\\')[:-1]
cwd = '/'.join(cwd)

DEBUG_MODE = False

class VoxelEngine:
    def __init__(self, _win_size=(1200, 800)):
        # * init pygame modules
        pg.init()
        
        # -------- settings -------- #
        # * window
        self.WIN_SIZE = _win_size
        self.BG_COLOR = (25/255, 25/255, 28/255)
        # * camera
        self.ASPECT_RATIO = _win_size[0]/_win_size[1]
        self.FOV_DEG = 50
        self.V_FOV = glm.radians(self.FOV_DEG)
        self.H_FOV = 2*math.atan(math.tan(self.V_FOV*0.5)*self.ASPECT_RATIO)
        self.NEAR = 0.1
        self.FAR = 2000
        self.MAX_PITCH = glm.radians(89)
        # * player
        self.PLAYER_SPEED = 0.0015
        self.PLAYER_ROT_SPEED = 0.004
        self.PLAYER_POS = glm.vec3(0, -3, 0)
        self.MOUSE_SENS = 0.002
        self.CREATIVE_MODE = True
        self.SHOW_WIRE = False
        self.SHOW_AXIS = False
        self.SHOW_VECT = False
        # * gui
        self.UIMANAGER = gui.UIManager(_win_size, 'themes/gui_theme.json')
        self.UIMANAGER.preload_fonts([{'name': 'fira_code', 'point_size': 24, 'style': 'bold_italic'}])
        self.SURF = pg.Surface(_win_size, pg.SRCALPHA)
        self.SURF.fill((0, 0, 0, 0))
        self.GUI_ELEMENTS = []
        
        # * interaction
        self.my_voxel = None
        # -------------------------- #
        
        # * get list of models
        self.models = next(os.walk(f'{cwd}/models/'))[1]
        self.curr_model = self.models[0]
        
        # * set opengl attributes
        pg.display.gl_set_attribute(pg.GL_CONTEXT_MAJOR_VERSION, 3)
        pg.display.gl_set_attribute(pg.GL_CONTEXT_MINOR_VERSION, 3)
        pg.display.gl_set_attribute(pg.GL_CONTEXT_PROFILE_MASK, pg.GL_CONTEXT_PROFILE_CORE)
        
        # * create opengl context
        self.WINDOW = pg.display.set_mode(self.WIN_SIZE, flags=pg.OPENGL|pg.DOUBLEBUF)
        self.UIMANAGER.set_visual_debug_mode(DEBUG_MODE)
        
        # * use opengl context
        self.ctx = mgl.create_context()
        self.ctx.front_face = 'cw'
        self.ctx.enable(flags=mgl.DEPTH_TEST|mgl.CULL_FACE)
        self.ctx.line_width = 4
        
        # * create clock to track time
        self.clock = pg.time.Clock()
        self.delta_time = 0
        self.time = 0
        
        # * init player and cube
        self.player = Player(self)
        self.cube = Cube(self)
        self.axis = Axis(self)
        self.crosshair = Crosshair(self)
        self.wireframe = WireFrame(self)
        self.gui = Gui(self)
        
        self.my_vector = (None, None)
        self.my_voxel = None
        
        self.vector = Vector(self)
        self.voxel = Voxel(self)
        
        # * init simulator
        self.sim = None
        def init_sim(_app):
            _app.sim = NCASimulator(_app.curr_model)
        threading.Thread(target=init_sim, args=[self]).start()
        
        # * hide mouse cursor
        pg.event.set_grab(True)
        pg.mouse.set_visible(False)
        
        # create gui elements
        w = self.WIN_SIZE[0]
        h = self.WIN_SIZE[1]
        half_w = w/2
        half_h = h/2

        # * current mode label
        self.mode_text = gui.elements.UILabel(relative_rect=pg.Rect((w-256, h-32), (256, 32)),
                                              text='current mode: free cam',
                                              manager=self.UIMANAGER,
                                              object_id=obj(object_id='#current_mode_label'))
        # * creative mode button
        self.enter_creative_button = gui.elements.UIButton(relative_rect=pg.Rect((260, 0), (w-260, h)),
                                                           text='',
                                                           manager=self.UIMANAGER,
                                                           object_id=obj(object_id='#invisible_button'))
        self.enter_creative_button.disable()
        self.GUI_ELEMENTS.append(self.enter_creative_button)
        
        # * toggle axis
        self.toggle_axis = gui.elements.UIButton(relative_rect=pg.Rect((4, 4+32+4), (256, 32)),
                                                 text='toggle axis: False',
                                                 manager=self.UIMANAGER)
        self.toggle_axis.disable()
        self.GUI_ELEMENTS.append(self.toggle_axis)

        # * toggle wireframe border
        self.toggle_border = gui.elements.UIButton(relative_rect=pg.Rect((4, 4+32+4+32+4), (256, 32)),
                                                 text='toggle border: False',
                                                 manager=self.UIMANAGER)
        self.toggle_border.disable()
        self.GUI_ELEMENTS.append(self.toggle_border)
        
        # * toggle click vector
        self.toggle_vector = gui.elements.UIButton(relative_rect=pg.Rect((4, 4+32+4+32+4+32+4), (256, 32)),
                                                 text='toggle vector: False',
                                                 manager=self.UIMANAGER)
        self.toggle_vector.disable()
        self.GUI_ELEMENTS.append(self.toggle_vector)
        
        # * toggle axis
        self.render_mode = gui.elements.UIButton(relative_rect=pg.Rect((4, 4+32+4+32+4+32+4+32+4), (256, 32)),
                                                 text='render mode: voxel',
                                                 manager=self.UIMANAGER)
        self.render_mode.disable()
        self.GUI_ELEMENTS.append(self.render_mode)
        
        # * model dropdown menu
        self.model_select = gui.elements.UIDropDownMenu(self.models, self.curr_model,
                                                        relative_rect=pg.Rect((4, 4), (256, 32)),
                                                        manager=self.UIMANAGER)
        self.model_select.disable()
        self.GUI_ELEMENTS.append(self.model_select)

        # * game is running
        self.is_running = True
    
    def fire_raycast(self):
        if self.sim != None:
            # * fire raycast from mouse pos through volume
            pos = self.player.pos
            vec = self.player.forward
            voxel = self.sim.raycast_volume(pos, vec)
            if voxel != None:
                self.my_voxel = voxel
            else:
                self.my_voxel = [1e12, 1e12, 1e12]
        
    def update(self):
        # * calculate fps
        self.delta_time = self.clock.tick()
        self.time = pg.time.get_ticks() * 0.001
        pg.display.set_caption(f'fps: {self.clock.get_fps() :.0f}')
        
        # * update gui
        self.UIMANAGER.update(self.delta_time)
        self.SURF.fill((0, 0, 0, 0))
        self.UIMANAGER.draw_ui(self.SURF)
        self.gui.update(self.SURF)
        
        # * update player
        if self.CREATIVE_MODE:
            self.player.update()
            
        # * update objects
        if self.SHOW_WIRE:
            self.wireframe.update()
        if self.SHOW_AXIS:
            self.axis.update()
        if self.SHOW_VECT:
            self.vector.update()
            
        # * update voxel
        if self.my_voxel != None and self.sim != None:
            self.voxel.update()
            
        self.vector.update()
        self.cube.update()
        
    def render(self):
        # * clear framebuffer
        self.ctx.clear(color=self.BG_COLOR)
        
        # * objects
        self.cube.render()
        if self.SHOW_WIRE:
            self.wireframe.render()
        if self.SHOW_AXIS:
            self.axis.render()
        if self.SHOW_VECT:
            self.vector.render()
            
        self.voxel.render()
        self.crosshair.render()
        self.gui.render()
        
        # * swap buffers
        pg.display.flip()
        
    def load_model(self, _model_name):
        # make sure sim is initialized
        if self.sim != None:
            # make sure model is different and exists
            if _model_name != self.curr_model and _model_name in self.models:
                if self.sim.unload():
                    # * set new model
                    self.curr_model = _model_name
                    self.cube.destroy()
                    self.cube = Cube(self)
                    def init_sim(_app):
                        _app.sim = NCASimulator(_app.curr_model)
                    threading.Thread(target=init_sim, args=[self]).start()

        
    def handle_events(self):
        for event in pg.event.get():
            # * quit application
            if event.type == pg.QUIT:
                self.is_running = False
                
            # ----------- gui events ----------- #
            self.UIMANAGER.process_events(event)
            
            if event.type == gui.UI_DROP_DOWN_MENU_CHANGED:
                if event.ui_element == self.model_select:
                    self.load_model(event.text)
                    self.model_select.disable()
                    def select_delay(_app):
                        while not _app.sim.is_loaded:
                            time.sleep(1)
                        if not self.CREATIVE_MODE:
                            self.model_select.enable()
                    threading.Thread(target=select_delay, args=[self]).start()
                    
            if event.type == gui.UI_BUTTON_PRESSED:
                if event.ui_element == self.enter_creative_button:
                    if not self.CREATIVE_MODE:
                        self.CREATIVE_MODE = True
                        self.mode_text.set_text('current mode: free cam')
                        # * hide mouse cursor
                        pg.event.set_grab(True)
                        pg.mouse.set_visible(False)
                        # * disable gui elements
                        for e in self.GUI_ELEMENTS:
                            e.disable()
                            
                if event.ui_element == self.toggle_axis:
                    self.SHOW_AXIS = not self.SHOW_AXIS
                    self.toggle_axis.set_text(f'toggle axis: {self.SHOW_AXIS}')
                    
                if event.ui_element == self.toggle_border:
                    self.SHOW_WIRE = not self.SHOW_WIRE
                    self.toggle_border.set_text(f'toggle axis: {self.SHOW_WIRE}')
                    
                if event.ui_element == self.toggle_vector:
                    self.SHOW_VECT = not self.SHOW_VECT
                    self.toggle_vector.set_text(f'toggle vector: {self.SHOW_VECT}')
                    
                if event.ui_element == self.render_mode:
                    if self.cube.toggle_blend():
                        self.render_mode.set_text(f'render mode: blend')
                    else:
                        self.render_mode.set_text(f'render mode: voxel')
                       
            # ---------------------------------- #
            
            
            # -------- key press events -------- #
            if event.type == pg.KEYDOWN:
            
                # * exit creative mode if press ESC key
                if event.key == pg.K_ESCAPE:
                    if self.CREATIVE_MODE:
                        self.CREATIVE_MODE = False
                        self.mode_text.set_text('current mode: gui')
                        # * show mouse cursor
                        pg.event.set_grab(False)
                        pg.mouse.set_visible(True)
                        # * enable gui elements
                        for e in self.GUI_ELEMENTS:
                            e.enable()
                        
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
                        
                # * rotate seed xy
                if event.key == pg.K_UP:
                    if self.sim != None:
                        self.sim.rot_seed('x')
                        
                # * rotate seed xz
                if event.key == pg.K_LEFT:
                    if self.sim != None:
                        self.sim.rot_seed('y')
                        
                # * rotate seed xz
                if event.key == pg.K_RIGHT:
                    if self.sim != None:
                        self.sim.rot_seed('z')
                    
            # ---------------------------------- #
                    
                    
            # ---------- mouse events ---------- #
            if pg.mouse.get_pressed(3)[0]:
                # * enter creative mode if click on screen
                if not self.CREATIVE_MODE:
                    pass
                    # TODO fix this so it works with gui interactions
                    # self.CREATIVE_MODE = True
                    # # * hide mouse cursor
                    # pg.event.set_grab(True)
                    # pg.mouse.set_visible(False)
                else:
                    if self.my_voxel != None:
                        if self.sim != None:
                            self.sim.erase_sphere(self.my_voxel, 6)
                            self.my_vector = (glm.vec3(self.player.pos), glm.normalize(glm.vec3(self.player.forward)))
            # ---------------------------------- #
            
    def run(self):
        while self.is_running:
            self.fire_raycast()
            self.handle_events()
            self.update()
            self.render()
            
        # * destroy scene
        self.cube.destroy()
        self.wireframe.destroy()
        self.axis.destroy()
        self.voxel.destroy()
        self.vector.destroy()
        self.gui.destroy()
        
        # * quit application
        print ('exiting application...')
        pg.quit()
        sys.exit()

        
if __name__ == '__main__':
    if DEBUG_MODE:
        print ('initializing in DEBUG MODE...')
    app = VoxelEngine()
    app.run()