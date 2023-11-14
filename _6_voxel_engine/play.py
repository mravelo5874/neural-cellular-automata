import moderngl as mgl
import pygame as pg
import sys

class VoxelEngine:
    def __init__(self, _win_size=(1600, 900)):
        # * init pygame modules
        pg.init()
        
        # * options
        self.WIN_SIZE = _win_size
        self.BG_COLOR = (1.0, 1.0, 1.0)
        
        # * set opengl attributes
        pg.display.gl_set_attribute(pg.GL_CONTEXT_MAJOR_VERSION, 3)
        pg.display.gl_set_attribute(pg.GL_CONTEXT_MINOR_VERSION, 3)
        pg.display.gl_set_attribute(pg.GL_CONTEXT_PROFILE_MASK, pg.GL_CONTEXT_PROFILE_CORE)
        
        # * create opengl context
        pg.display.set_mode(self.WIN_SIZE, flags=pg.OPENGL|pg.DOUBLEBUF)
        
        # * use opengl context
        self.ctx = mgl.create_context()
        
        # * create clock to track time
        self.clock = pg.time.Clock()
        self.delta_time = 0
        self.time = 0
        
        # * game is running
        self.is_running = True
        
    def update(self):
        # * calculate fps
        self.delta_time = self.clock.tick()
        self.time = pg.time.get_ticks() * 0.001
        pg.display.set_caption(f'fps: {self.clock.get_fps() :.0f}')
        
    def render(self):
        # * clear framebuffer
        self.ctx.clear(color=self.BG_COLOR)
        
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
        pg.quit()
        sys.exit()

        
if __name__ == '__main__':
    app = VoxelEngine()
    app.run()