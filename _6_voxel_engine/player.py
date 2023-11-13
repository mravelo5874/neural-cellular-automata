import pygame as pg
from camera import Camera
from settings import *

class Player(Camera):
    def __init__(self, _app, _pos=PLAYER_INIT_POS, _yaw=-90, _pitch=0):
        self.app = _app
        super().__init__(_pos, _yaw, _pitch)
        
    def update(self):
        self.keyboard_control()
        self.mouse_control()
        super().update()
    
    def mouse_control(self):
        mouse_dx, mouse_dy = pg.mouse.get_rel()
        if mouse_dx:
            self.rotate_yaw(_delta_x=mouse_dx*MOUSE_SENS)
        if mouse_dy:
            self.rotate_pitch(_delta_y=mouse_dy*MOUSE_SENS)
    
    def keyboard_control(self):
        state = pg.key.get_pressed()
        vel = PLAYER_SPEED * self.app.delta_time
        # * control player movement
        if state[pg.K_w]:
            self.move_forward(vel)
        if state[pg.K_s]:
            self.move_back(vel)
        if state[pg.K_d]:
            self.move_right(vel)
        if state[pg.K_a]:
            self.move_left(vel)
        if state[pg.K_SPACE]:
            self.move_up(vel)
        if state[pg.K_LSHIFT]:
            self.move_down(vel)        