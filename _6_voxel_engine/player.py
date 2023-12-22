import pygame as pg
from camera import Camera

class Player(Camera):
    def __init__(self, _app, _yaw=-0.8, _pitch=-0.7):
        self.app = _app
        self.prev_mouse = pg.mouse.get_pos()
        super().__init__(_app, _app.PLAYER_POS, _yaw, _pitch)
        
    def update(self):
        self.keyboard_control()
        self.mouse_control()
        super().update()
        
    def mouse_control(self):
        mdx, mdy = pg.mouse.get_rel()
        if mdx:
            self.rotate_yaw(mdx * self.app.MOUSE_SENS)
        if mdy:
            self.rotate_pitch(mdy * self.app.MOUSE_SENS)

    def keyboard_control(self):
        state = pg.key.get_pressed()
        vel = self.app.PLAYER_SPEED * self.app.delta_time
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