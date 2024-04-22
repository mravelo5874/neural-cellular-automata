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
from plane import Plane
from gui import Gui
from nca_simulator import NCASimulator

cwd = os.getcwd().split('\\')[:-1]
cwd = '/'.join(cwd)

DEBUG_MODE = False

class VoxelEngine:
    def __init__(self, _win_size=(1200, 800)): # * small window: (900, 450), large window: (1200, 800)
        # * init pygame modules
        pg.init()
        pg.display.set_caption('Neural Cellular Automata Renderer')
        
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
        self.PLAYER_POS = glm.vec3(1.4, -1.4, 1.4)
        self.MOUSE_SENS = 0.002
        self.CREATIVE_MODE = False
        self.SHOW_WIRE = False
        self.SHOW_AXIS = False
        self.SHOW_VECT = False
        self.SHOW_PLANE = False
        self.SEND_RAYCASTS = False
        # * gui
        self.UIMANAGER = gui.UIManager(_win_size, 'themes/gui_theme.json')
        self.UIMANAGER.preload_fonts([{'name': 'fira_code', 'point_size': 24, 'style': 'bold_italic'}])
        self.SURF = pg.Surface(_win_size, pg.SRCALPHA)
        self.SURF.fill((0, 0, 0, 0))
        self.GUI_ELEMENTS = []
        self.PLANE_GUI_ELEMENTS = []
        # * slice plane
        self.plane_pos = [0.0, 0.0, 0.0]
        self.plane_rot = [0.0, 0.0, 0.0] # (PI/4, PI/4, PI/4)
        
        # * interaction
        self.my_vector = (None, None)
        self.my_voxel = [1e12, 1e12, 1e12]
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
        
        self.plane = Plane(self)
        self.vector = Vector(self)
        self.voxel = Voxel(self)
        
        # * init simulator
        self.sim = None
        def init_sim(_app):
            _app.sim = NCASimulator(_app.curr_model, self)
        threading.Thread(target=init_sim, args=[self]).start()
        
        # create gui elements
        w = self.WIN_SIZE[0]
        h = self.WIN_SIZE[1]
        half_w = w/2
        half_h = h/2
        
        # * paused label
        self.paused_text = gui.elements.UILabel(relative_rect=pg.Rect((260, 18), (256, 32)),
                                              text='',
                                              manager=self.UIMANAGER,
                                              object_id=obj(object_id='#label_left'))
        # * iterations label
        self.iter_text = gui.elements.UILabel(relative_rect=pg.Rect((260, 0), (256, 32)),
                                              text='0',
                                              manager=self.UIMANAGER,
                                              object_id=obj(object_id='#label_left'))
        #self.iter_text.set_tooltip('The number of iterations the model has performed. One iteration is a single forward function execution.')
        # * current mode label
        self.mode_text = gui.elements.UILabel(relative_rect=pg.Rect((w-512, 0), (512, 32)),
                                              text='gui',
                                              manager=self.UIMANAGER,
                                              object_id=obj(object_id='#label_right'))
        #self.mode_text.set_tooltip('If in free cam mode, you can fly around using WASD, Space, and LShift. If in gui mode, you may interact with the gui.')
        # * current position label
        p = self.player.pos
        pos = '({:.3f}, {:.3f}, {:.3f})'.format(p[0], p[1], p[2])
        self.pos_text = gui.elements.UILabel(relative_rect=pg.Rect((w-256, h-32), (256, 32)),
                                              text=f'{pos}', 
                                              manager=self.UIMANAGER,
                                              object_id=obj(object_id='#label_right'))
        #self.pos_text.set_tooltip('Your current position in space.')
        # * current mode label
        self.fps_text = gui.elements.UILabel(relative_rect=pg.Rect((0, h-32), (256, 32)),
                                              text='',
                                              manager=self.UIMANAGER,
                                              object_id=obj(object_id='#label_left'))
        #self.fps_text.set_tooltip('The number of frames rendered per second.')
        # * creative mode button
        self.enter_creative_button = gui.elements.UIButton(relative_rect=pg.Rect((260, 0), (w-260, h)),
                                                           text='',
                                                           manager=self.UIMANAGER,
                                                           object_id=obj(object_id='#invisible_button'))
        self.GUI_ELEMENTS.append(self.enter_creative_button)
        # * toggle axis
        self.toggle_axis = gui.elements.UIButton(relative_rect=pg.Rect((4, 64+4+32+4), (256, 32)),
                                                 text='toggle axis: _',
                                                 manager=self.UIMANAGER)
        self.GUI_ELEMENTS.append(self.toggle_axis)
        #self.toggle_axis.set_tooltip('Toggle to render axis lines.')

        # * toggle wireframe border
        self.toggle_border = gui.elements.UIButton(relative_rect=pg.Rect((4, 64+4+32+4+32+4), (256, 32)),
                                                 text='toggle border: _',
                                                 manager=self.UIMANAGER)
        self.GUI_ELEMENTS.append(self.toggle_border)
        #self.toggle_border.set_tooltip('Toggle to render a wire-frame border around the volume.')
        
        # * toggle click vector
        self.toggle_vector = gui.elements.UIButton(relative_rect=pg.Rect((4, 64+4+32+4+32+4+32+4), (256, 32)),
                                                 text='toggle vector: _',
                                                 manager=self.UIMANAGER)
        self.GUI_ELEMENTS.append(self.toggle_vector)
        #self.toggle_vector.set_tooltip('Toggle to render a vector whenever the user clicks in free cam mode.')
        
        # * toggle render modes
        self.render_mode = gui.elements.UIButton(relative_rect=pg.Rect((4, 64+4+32+4+32+4+32+4+32+4), (256, 32)),
                                                 text='render mode: _',
                                                 manager=self.UIMANAGER, )
        self.GUI_ELEMENTS.append(self.render_mode)
        #self.render_mode.set_tooltip('Toggle between rendering voxels or blended.')
        
        # * toggle raycast volume
        self.raycast_vol = gui.elements.UIButton(relative_rect=pg.Rect((4, 64+4+32+4+32+4+32+4+32+4+32+4), (256, 32)),
                                                 text='raycast vol: _',
                                                 manager=self.UIMANAGER, )
        self.GUI_ELEMENTS.append(self.raycast_vol)
        #self.raycast_vol.set_tooltip('Toggle raycasting the volume in order to damage the automata.')
        
        # * plane label text
        self.slice_plane_label = gui.elements.UILabel(relative_rect=pg.Rect((4, 64+4+32+4+32+4+32+4+32+4+32+4+32+4), (256, 32)),
                                                 text='slice plane opts',
                                                 manager=self.UIMANAGER, 
                                                 object_id=obj(object_id='#label_left'))
        self.GUI_ELEMENTS.append(self.slice_plane_label)
        
        # * toggle plane
        self.toggle_plane = gui.elements.UIButton(relative_rect=pg.Rect((4, 64+4+32+4+32+4+32+4+32+4+32+4+32+4+32+4), (256, 32)),
                                                 text='slice place: _',
                                                 manager=self.UIMANAGER, )
        self.GUI_ELEMENTS.append(self.toggle_plane)
        #self.toggle_plane.set_tooltip('Toggle raycasting the volume in order to damage the automata.')
        
        # * plane position label text
        self.slice_plane_pos_label = gui.elements.UILabel(relative_rect=pg.Rect((4, 64+4+32+4+32+4+32+4+32+4+32+4+32+4+32+4+32+4), (256, 32)),
                                                 text='position',
                                                 manager=self.UIMANAGER, 
                                                 object_id=obj(object_id='#label_left'))
        self.GUI_ELEMENTS.append(self.slice_plane_pos_label)
        
        # * plane x pos label
        self.plane_pos_x_label = gui.elements.UILabel(relative_rect=pg.Rect((4, 64+4+32+4+32+4+32+4+32+4+32+4+32+4+32+4+32+4+32+4), (90, 32)),
                                                  text=f'x: {round(self.plane_pos[0], 2)}',
                                                  manager=self.UIMANAGER, 
                                                  object_id=obj(object_id='#label_left'))
        self.GUI_ELEMENTS.append(self.plane_pos_x_label)
        
        # * plane x pos slider
        self.plane_pos_x_slider = gui.elements.UIHorizontalSlider(relative_rect=pg.Rect((90+4, 64+4+32+4+32+4+32+4+32+4+32+4+32+4+32+4+32+4+32+4), (166, 32)),
                                                 start_value=self.plane_pos[0],
                                                 value_range=[-2.0, 2.0],
                                                 click_increment=0.01,
                                                 manager=self.UIMANAGER, )
        self.PLANE_GUI_ELEMENTS.append(self.plane_pos_x_slider)
        
        # * plane y pos label
        self.plane_pos_y_label = gui.elements.UILabel(relative_rect=pg.Rect((4, 64+4+32+4+32+4+32+4+32+4+32+4+32+4+32+4+32+4+32+4+32+4), (90, 32)),
                                                  text=f'y: {round(self.plane_pos[1], 2)}',
                                                  manager=self.UIMANAGER, 
                                                  object_id=obj(object_id='#label_left'))
        self.GUI_ELEMENTS.append(self.plane_pos_y_label)
        
        # * plane y pos slider
        self.plane_pos_y_slider = gui.elements.UIHorizontalSlider(relative_rect=pg.Rect((90+4, 64+4+32+4+32+4+32+4+32+4+32+4+32+4+32+4+32+4+32+4+32+4), (166, 32)),
                                                 start_value=self.plane_pos[1],
                                                 value_range=[-2.0, 2.0],
                                                 click_increment=0.01,
                                                 manager=self.UIMANAGER, )
        self.PLANE_GUI_ELEMENTS.append(self.plane_pos_y_slider)
        
        # * plane z pos label
        self.plane_pos_z_label = gui.elements.UILabel(relative_rect=pg.Rect((4, 64+4+32+4+32+4+32+4+32+4+32+4+32+4+32+4+32+4+32+4+32+4+32+4), (90, 32)),
                                                  text=f'z: {round(self.plane_pos[2], 2)}',
                                                  manager=self.UIMANAGER, 
                                                  object_id=obj(object_id='#label_left'))
        self.GUI_ELEMENTS.append(self.plane_pos_z_label)
        
        # * plane z pos slider
        self.plane_pos_z_slider = gui.elements.UIHorizontalSlider(relative_rect=pg.Rect((90+4, 64+4+32+4+32+4+32+4+32+4+32+4+32+4+32+4+32+4+32+4+32+4+32+4), (166, 32)),
                                                 start_value=self.plane_pos[2],
                                                 value_range=[-2.0, 2.0],
                                                 click_increment=0.01,
                                                 manager=self.UIMANAGER, )
        self.PLANE_GUI_ELEMENTS.append(self.plane_pos_z_slider)
        
         # * plane rotation label text
        self.slice_plane_rot_label = gui.elements.UILabel(relative_rect=pg.Rect((4, 64+4+32+4+32+4+32+4+32+4+32+4+32+4+32+4+32+4+32+4+32+4+32+4+32+4), (256, 32)),
                                                 text='rotation',
                                                 manager=self.UIMANAGER, 
                                                 object_id=obj(object_id='#label_left'))
        self.GUI_ELEMENTS.append(self.slice_plane_rot_label)
        
        # * plane x rot label
        self.plane_rot_x_label = gui.elements.UILabel(relative_rect=pg.Rect((4, 64+4+32+4+32+4+32+4+32+4+32+4+32+4+32+4+32+4+32+4+32+4+32+4+32+4+32+4), (90, 32)),
                                                  text=f'x: {round(self.plane_rot[0], 2)}π',
                                                  manager=self.UIMANAGER, 
                                                  object_id=obj(object_id='#label_left'))
        self.GUI_ELEMENTS.append(self.plane_rot_x_label)
        
        # * plane x rot slider
        self.plane_rot_x_slider = gui.elements.UIHorizontalSlider(relative_rect=pg.Rect((90+4, 64+4+32+4+32+4+32+4+32+4+32+4+32+4+32+4+32+4+32+4+32+4+32+4+32+4+32+4), (166, 32)),
                                                 start_value=self.plane_rot[0],
                                                 value_range=[0.0, 2.0],
                                                 click_increment=0.01,
                                                 manager=self.UIMANAGER, )
        self.PLANE_GUI_ELEMENTS.append(self.plane_rot_x_slider)
        
        # * plane y rot label
        self.plane_rot_y_label = gui.elements.UILabel(relative_rect=pg.Rect((4, 64+4+32+4+32+4+32+4+32+4+32+4+32+4+32+4+32+4+32+4+32+4+32+4+32+4+32+4+32+4), (90, 32)),
                                                  text=f'y: {round(self.plane_rot[1], 2)}π',
                                                  manager=self.UIMANAGER, 
                                                  object_id=obj(object_id='#label_left'))
        self.GUI_ELEMENTS.append(self.plane_rot_y_label)
        
        # * plane y rot slider
        self.plane_rot_y_slider = gui.elements.UIHorizontalSlider(relative_rect=pg.Rect((90+4, 64+4+32+4+32+4+32+4+32+4+32+4+32+4+32+4+32+4+32+4+32+4+32+4+32+4+32+4+32+4), (166, 32)),
                                                 start_value=self.plane_rot[1],
                                                 value_range=[0.0, 2.0],
                                                 click_increment=0.01,
                                                 manager=self.UIMANAGER, )
        self.PLANE_GUI_ELEMENTS.append(self.plane_rot_y_slider)
        
        # * plane z rot label
        self.plane_rot_z_label = gui.elements.UILabel(relative_rect=pg.Rect((4, 64+4+32+4+32+4+32+4+32+4+32+4+32+4+32+4+32+4+32+4+32+4+32+4+32+4+32+4+32+4+32+4), (90, 32)),
                                                  text=f'z: {round(self.plane_rot[2], 2)}π',
                                                  manager=self.UIMANAGER, 
                                                  object_id=obj(object_id='#label_left'))
        self.GUI_ELEMENTS.append(self.plane_rot_z_label)
        
        # * plane z rot slider
        self.plane_rot_z_slider = gui.elements.UIHorizontalSlider(relative_rect=pg.Rect((90+4, 64+4+32+4+32+4+32+4+32+4+32+4+32+4+32+4+32+4+32+4+32+4+32+4+32+4+32+4+32+4+32+4), (166, 32)),
                                                 start_value=self.plane_rot[2],
                                                 value_range=[0.0, 2.0],
                                                 click_increment=0.01,
                                                 manager=self.UIMANAGER, )
        self.PLANE_GUI_ELEMENTS.append(self.plane_rot_z_slider)
        
        # * model dropdown menu
        self.model_select = gui.elements.UIDropDownMenu(self.models, self.curr_model,
                                                        relative_rect=pg.Rect((4, 4), (256, 32)),
                                                        manager=self.UIMANAGER,
                                                        object_id=obj(object_id='#drop_down'))
        self.GUI_ELEMENTS.append(self.model_select)
        self.model_select.set_tooltip('Select a model to render.')
        
        # * reset button
        self.reset_button = gui.elements.UIButton(relative_rect=pg.Rect((4, 4+32+4), (82, 32)),
                                                  text='reset',
                                                  manager=self.UIMANAGER,
                                                  object_id=obj(object_id='#reset_button'))
        self.GUI_ELEMENTS.append(self.reset_button)
        self.reset_button.set_tooltip('Reset to the model\'s starting seed.')
        
        # * pause button
        self.pause_button = gui.elements.UIButton(relative_rect=pg.Rect((4+82+4, 4+32+4), (82, 32)),
                                                  text='pause',
                                                  manager=self.UIMANAGER,
                                                  object_id=obj(object_id='#pause_button'))
        self.GUI_ELEMENTS.append(self.pause_button)
        self.pause_button.set_tooltip('Pause the model\'s execution.')
        
        # * step button
        self.step_button = gui.elements.UIButton(relative_rect=pg.Rect((4+82+4+82+4, 4+32+4), (83, 32)),
                                                  text='step',
                                                  manager=self.UIMANAGER,
                                                  object_id=obj(object_id='#step_button'))
        self.GUI_ELEMENTS.append(self.step_button)
        self.step_button.set_tooltip('If paused, perform a single update step.')
        
        # * enable/disable plane gui options
        if self.SHOW_PLANE:
            for e in self.PLANE_GUI_ELEMENTS:
                e.enable()
        else:
            for e in self.PLANE_GUI_ELEMENTS:
                e.disable()

        self.toggle_axis.set_text(f'toggle axis: {self.SHOW_AXIS}', )
        self.toggle_border.set_text(f'toggle border: {self.SHOW_WIRE}')
        self.toggle_vector.set_text(f'toggle vector: {self.SHOW_VECT}')
        self.raycast_vol.set_text(f'raycast vol: {self.SEND_RAYCASTS}')
        self.toggle_plane.set_text(f'slice plane: {self.SHOW_PLANE}')
        self.render_mode.set_text('render mode: voxel')

        # * game is running
        self.is_running = True
        
    def update(self):
        # * calculate fps
        self.delta_time = self.clock.tick()
        self.time = pg.time.get_ticks() * 0.001
        self.fps_text.set_text(f'{self.clock.get_fps() :.0f}')

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
        if self.SHOW_PLANE:
            self.plane.update()
            
        # * update voxel
        if self.my_voxel != None and self.sim != None:
            if self.SEND_RAYCASTS:
                self.voxel.update()
            
        # * update pos text
        p = self.player.pos
        pos = '({:.3f}, {:.3f}, {:.3f})'.format(p[0], p[1], p[2])
        self.pos_text.set_text(f'{pos}')
        
        # * update iteration text
        if self.sim != None:
            self.iter_text.set_text(f'{self.sim.iter}')
            
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
        if self.SEND_RAYCASTS:
            self.voxel.render()
        if self.SHOW_PLANE:
            self.plane.render()
            
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
                        _app.sim = NCASimulator(_app.curr_model, self)
                        _app.render_mode.set_text(f'render mode: voxel')
                        _app.pause_button.set_text('pause')
                        _app.paused_text.set_text('')
                    threading.Thread(target=init_sim, args=[self]).start()

        
    def handle_events(self):
        for event in pg.event.get():
            # * quit application
            if event.type == pg.QUIT:
                self.is_running = False
                
            # ----------- gui events ----------- #
            self.UIMANAGER.process_events(event)
            
            # * check for gui button presses
            if event.type == gui.UI_HORIZONTAL_SLIDER_MOVED:
                if event.ui_element == self.plane_pos_x_slider:
                    self.plane_pos[0] = event.value
                    self.plane_pos_x_label.set_text(f'x: {round(self.plane_pos[0], 2)}')
                    
                if event.ui_element == self.plane_pos_y_slider:
                    self.plane_pos[1] = event.value
                    self.plane_pos_y_label.set_text(f'y: {round(self.plane_pos[1], 2)}')
                    
                if event.ui_element == self.plane_pos_z_slider:
                    self.plane_pos[2] = event.value
                    self.plane_pos_z_label.set_text(f'z: {round(self.plane_pos[2], 2)}')
                    
                if event.ui_element == self.plane_rot_x_slider:
                    self.plane_rot[0] = event.value
                    self.plane_rot_x_label.set_text(f'x: {round(self.plane_rot[0], 1)}π')
                    
                if event.ui_element == self.plane_rot_y_slider:
                    self.plane_rot[1] = event.value
                    self.plane_rot_y_label.set_text(f'y: {round(self.plane_rot[1], 1)}π')
                    
                if event.ui_element == self.plane_rot_z_slider:
                    self.plane_rot[2] = event.value
                    self.plane_rot_z_label.set_text(f'z: {round(self.plane_rot[2], 1)}π')
            
            # * load in new model from drop-down menu
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

            # * check for gui button presses
            if event.type == gui.UI_BUTTON_PRESSED:
                if event.ui_element == self.enter_creative_button:
                    if not self.CREATIVE_MODE:
                        self.CREATIVE_MODE = True
                        self.mode_text.set_text('free cam')
                        self.enter_creative_button.set_text('')
                        # * hide mouse cursor
                        pg.event.set_grab(True)
                        pg.mouse.set_visible(False)
                        # * reset mouse position
                        pg.mouse.set_pos(self.WIN_SIZE[0]/2, self.WIN_SIZE[1]/2)
                        self.player.reset_mouse_rel()
                        # * disable gui elements
                        for e in self.GUI_ELEMENTS:
                            e.disable()
                        for e in self.PLANE_GUI_ELEMENTS:
                            e.disable()

                # * toggle axis lines           
                if event.ui_element == self.toggle_axis:
                    self.SHOW_AXIS = not self.SHOW_AXIS
                    self.toggle_axis.set_text(f'toggle axis: {self.SHOW_AXIS}')
                
                # * toggle wireframe border
                if event.ui_element == self.toggle_border:
                    self.SHOW_WIRE = not self.SHOW_WIRE
                    self.toggle_border.set_text(f'toggle border: {self.SHOW_WIRE}')
                
                # * toggle vector on click 
                if event.ui_element == self.toggle_vector:
                    self.SHOW_VECT = not self.SHOW_VECT
                    self.toggle_vector.set_text(f'toggle vector: {self.SHOW_VECT}')
                
                # * toggle render modes
                if event.ui_element == self.render_mode:
                    if self.cube.toggle_blend():
                        self.render_mode.set_text(f'render mode: blend')
                    else:
                        self.render_mode.set_text(f'render mode: voxel')
                        
                # * toggle raycast volume 
                if event.ui_element == self.raycast_vol:
                    self.SEND_RAYCASTS = not self.SEND_RAYCASTS
                    self.raycast_vol.set_text(f'raycast vol: {self.SEND_RAYCASTS}')
                    
                # * toggle slice plane
                if event.ui_element == self.toggle_plane:
                    self.SHOW_PLANE = not self.SHOW_PLANE
                    self.toggle_plane.set_text(f'slice plane: {self.SHOW_PLANE}')
                    
                    # * enable/disable plane gui options
                    if self.SHOW_PLANE:
                        for e in self.PLANE_GUI_ELEMENTS:
                            e.enable()
                    else:
                        for e in self.PLANE_GUI_ELEMENTS:
                            e.disable()
                
                # * reset current model
                if event.ui_element == self.reset_button:
                    if self.sim != None:
                        self.sim.reset()
                
                # * pause current model
                if event.ui_element == self.pause_button:
                    if self.sim != None:
                        self.sim.toggle_pause()
                        if self.sim.is_paused:
                            self.paused_text.set_text('[PAUSED]')
                            self.pause_button.set_text('play')
                        else:
                            self.paused_text.set_text('')
                            self.pause_button.set_text('pause')
                            
                # * step forward current model
                if event.ui_element == self.step_button:
                    if self.sim != None:
                        self.sim.step_forward()
            # ---------------------------------- #
            
            
            # -------- key press events -------- #
            if event.type == pg.KEYDOWN:
                # * exit creative mode if press ESC key
                if event.key == pg.K_ESCAPE:
                    if self.CREATIVE_MODE:
                        self.CREATIVE_MODE = False
                        self.mode_text.set_text('gui')
                        # * reset mouse position
                        pg.mouse.set_pos(self.WIN_SIZE[0]/2, self.WIN_SIZE[1]/2)
                        # * show mouse cursor
                        pg.event.set_grab(False)
                        pg.mouse.set_visible(True)
                        # * enable gui elements
                        for e in self.GUI_ELEMENTS:
                            e.enable()
                        if self.SHOW_PLANE:
                            for e in self.PLANE_GUI_ELEMENTS:
                                e.enable()
                        
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
                
                # * load custom seeds
                if event.key == pg.K_0:
                    if self.sim != None:
                        self.sim.load_custom(0) 
                if event.key == pg.K_1:
                    if self.sim != None:
                        self.sim.load_custom(1)
                if event.key == pg.K_2:
                    if self.sim != None:
                        self.sim.load_custom(2)
                if event.key == pg.K_3:
                    if self.sim != None:
                        self.sim.load_custom(3)
                if event.key == pg.K_4:
                    if self.sim != None:
                        self.sim.load_custom(4)
                if event.key == pg.K_5:
                    if self.sim != None:
                        self.sim.load_custom(5)
                if event.key == pg.K_6:
                    if self.sim != None:
                        self.sim.load_custom(6)
                    
            # ---------------------------------- #
                    
                    
            # ---------- mouse events ---------- #
            if pg.mouse.get_pressed(3)[0]:
                # * enter creative mode if click on screen
                if self.CREATIVE_MODE:
                    if self.my_voxel != None:
                        if self.sim != None:
                            if self.SEND_RAYCASTS:
                                self.sim.erase_sphere(self.my_voxel, 6)
                                self.my_vector = (glm.vec3(self.player.pos), glm.normalize(glm.vec3(self.player.forward)))
            # ---------------------------------- #
            
    def run(self):
        while self.is_running:
            self.handle_events()
            self.update()
            self.render()
        
        # * stop sim
        self.sim.stop()
        
        # * destroy scene
        self.cube.destroy()
        self.wireframe.destroy()
        self.axis.destroy()
        self.voxel.destroy()
        self.vector.destroy()
        self.gui.destroy()
        self.plane.destroy()
        
        # * quit application
        print ('exiting application...')
        pg.quit()
        sys.exit()

        
if __name__ == '__main__':
    if DEBUG_MODE:
        print ('initializing in DEBUG MODE...')
    app = VoxelEngine()
    app.run()