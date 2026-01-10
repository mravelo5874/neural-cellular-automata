from utility import *

from radial_automata import RadialAutomata
from decisional_nn import DecisionalNN

_WINDOW_BG_COLOR_ = (255, 255, 255)
_WINDOW_TEXT_COLOR_ = (0, 0, 0)
_SIZE_ = 512
_SCALE_ = 64
_AUTO_RUN_ = True

_RADIUS_ = 1.0
_RATE_ = 0.1
_NUM_ = 6
_BOUND_ = 3.0

# * sets the device  
_DEVICE_ = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.backends.cudnn.benchmark = True
torch.cuda.empty_cache()
if _DEVICE_ == 'cuda':
    torch.set_default_tensor_type('torch.cuda.FloatTensor')

pygame.init()
pygame.display.set_caption('radial automata simulation')

window = pygame.display.set_mode((_SIZE_, _SIZE_))
model = DecisionalNN(16, 19, _device=_DEVICE_)
automata = RadialAutomata(_RADIUS_, _RATE_, _NUM_, _BOUND_)

# * create clock for fps
clock = pygame.time.Clock()
delta_time = 0

# * setup text rendering
font_size = 20
my_font = pygame.font.SysFont('consolas', font_size)
fps_surface = my_font.render(f'{clock.get_fps() :.0f}', False, _WINDOW_TEXT_COLOR_)
cells_sureface = my_font.render(f'dots: {len(automata.cells)}', False, _WINDOW_TEXT_COLOR_)

# * mutex
mutex = threading.Lock()

running = True 
def run_sim(_delay):
    time.sleep(_delay)
    while running:
        mutex.acquire()
        automata.update(model)
        mutex.release()

# # * start forward worker
if _AUTO_RUN_:
    sim = threading.Thread(target=run_sim, args=[1], daemon=False)
    sim.start()

while running:
    # * close application
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                running = False
                break
            if event.key == pygame.K_RETURN:
                mutex.acquire()
                automata.update(model)
                mutex.release()
            if event.key == pygame.K_r:
                mutex.acquire()
                model = DecisionalNN(16, 19, _device=_DEVICE_)
                automata = RadialAutomata(_RADIUS_, _RATE_, _NUM_, _BOUND_)
                mutex.release()
            if event.key == pygame.K_p:
                image = automata.pixelize(2, 32)
                image = np.rot90(image, 1)
                plt.axis('off')
                plt.imshow(image)
                plt.show()
        
    # * draw tensor to window
    window.fill(_WINDOW_BG_COLOR_)
    
    # * blip cells
    cells = automata.cells

    for cell in cells:
        color = cell.color.rgba()
        rgb, a = color[:3], color[3:4]
        color = np.clip(1.0 - a + rgb, 0.0, 1.0)
        color = np.array(color * 255)
        color = np.clip(color, 0, 255).astype(int)
        cell_pos = cell.pos.xy()
        cell_pos[1] *= -1.0
        pos = (cell_pos*_SCALE_) + (_SIZE_/2, _SIZE_/2)
        pygame.draw.circle(window, color, pos[:2,], 5, 5) #(r, g, b) is color, (x, y) is center, R is radius and w is the thickness of the circle border.
    
    # * calculate fps
    delta_time = clock.tick()
    fps_surface = my_font.render(f'{clock.get_fps() :.0f}', False, _WINDOW_TEXT_COLOR_)
    window.blit(fps_surface, (0, 0))

    # * show number of cells
    cells_sureface = my_font.render(f'dots: {len(cells)}', False, _WINDOW_TEXT_COLOR_)
    window.blit(cells_sureface, (0, _SIZE_-font_size))
    
    # * flip it!
    pygame.display.flip()

# * quit it!
pygame.quit()

if _AUTO_RUN_:
    sim.join()