from utility import *
from radial_cell import RadialCell
from chunk_system import ChunkSystem

Action = Enum('Action', ['MOVE', 'MOD_COLOR'])

class RadialAutomataState:
    def __init__(self, _grid, _cells):
        self.grid = _grid
        self.cells = _cells

class RadialAutomata:
    def __init__(self, _radius, _rate, _num, _bound=4.0, _pmap_res=8, _color_scale=0.2, _move_scale=0.1, _cell_limit=2048, _neighbor_limit=32):
        self.radius = _radius
        self.rate = _rate
        self.num = _num
        
        self.bound=_bound
        self.pmap_res = _pmap_res
        self.color_scale = _color_scale
        self.move_scale = _move_scale
        self.cell_limit = _cell_limit
        self.neighbor_limit = _neighbor_limit
        self.chunk_size = _radius
        self.reset()

    def reset(self):
        self.grid = ChunkSystem(self.radius)
        self.cells = []

        # * init cells
        h = self.num//2
        d = 0.5
        cid = 0
        for x in range(self.num):
            for y in range(self.num):
                pos = np.array([(x-h)*d, (y-h)*d])
                color = np.array([0.5, 0.5, 0.5, 1.0])
                color[3] = 1.0
                angle = np.random.random() * np.pi * 2.0
                hidden = np.random.random(12)
                cell = RadialCell(pos, color, angle, hidden, cid, self.bound)
                self.grid.add_cell(cell)
                self.cells.append(cell)
                cid += 1
                
    def load_state(self, _state):
        self.grid = _state.grid
        self.cells = _state.cells
    
    def save_state(self):
        state = RadialAutomataState(self.grid, self.cells)
        return state

    def pixelize(self, _scale, _size):
        image = np.ones([_size, _size, 3]).astype(np.float32)
        num_cells = {}
        
        # * find center-most cell and use as origin
        cen = np.array([0.0, 0.0])
        for cell in self.cells:
            cen += cell.pos.xy()
        cen /= len(self.cells)
        
        for cell in self.cells:
            cell_pos = cell.pos.xy()
            pos = ((cell_pos - cen) * _scale) + (_size/2, _size/2)
            pos = pos.astype(int)
            if pos[0] < _size and pos[0] >= 0 and pos[1] < _size and pos[1] >= 0:
                color = cell.color.rgba()
                rgb, a = color[:3], color[3:4]
                color = np.clip(1.0 - a + rgb, 0.0, 1.0)
                # * blend color if pixel already colored
                key = str(pos)
                if key in num_cells:
                    image[pos[0], pos[1]] *= num_cells[key]
                    image[pos[0], pos[1]] += color
                    num_cells[key] += 1
                    image[pos[0], pos[1]] /= num_cells[key]
                else:
                    num_cells[key] = 1
                    image[pos[0], pos[1]] = color
        image = np.array(image * 255.0).astype(np.uint8)
        return image

    def update(self, _model):
        # * stochastic update
        stochastic_mask = np.random.rand(len(self.cells)) < self.rate
        active_cells =  np.argwhere(stochastic_mask).flatten()
        
        # * perception step
        perception, neighbors = self.percieve(active_cells)
        
        # * each active cell performs its action based on its perception
        self.perform(active_cells, perception, neighbors, _model)
        
    def percieve(self, _active_cells):
        perception = torch.zeros([len(_active_cells), 16, self.pmap_res*2+1, self.pmap_res*2+1])
        neighbors_count = []

        for i, c in enumerate(_active_cells):
            cell = self.cells[c]
            angle = cell.angle + (np.pi / 2.0)
            cos = np.cos(angle)
            sin = np.sin(angle)

            # * init perception map
            pmap = np.zeros([self.pmap_res*2+1, self.pmap_res*2+1, 16])
            pmap[self.pmap_res, self.pmap_res] = cell.state()

            # currate cells from current and adjacent chunks
            neighbors = self.grid.query_neighbors_in_radius(cell)
            for neighbor in neighbors:
                dir = neighbor.pos.xy() - cell.pos.xy()
                dir = dir / self.radius
                dir = np.array([cos*dir[0] -sin*dir[1], sin*dir[0]+cos*dir[1]])
                loc = np.floor(dir * self.pmap_res).astype(int)
                loc += self.pmap_res
                pmap[loc[0], loc[1]] = neighbor.state()

            # * add to overall perception
            x = torch.tensor(pmap, dtype=torch.float32).permute(2, 0, 1)
            perception[i] = x
            neighbors_count.append(len(neighbors))

            # * show perception map
            # print (f'angle: {cell.angle}')
            # pmap = pmap.transpose([2, 0, 1])
            # rgb, a = pmap[:3], pmap[3:4]
            # color = np.clip(1.0 - a + rgb, 0.0, 1.0)
            # image = (color*255.0).astype(np.uint8)
            # image = image.transpose([1, 2, 0])
            # plt.axis('off')
            # plt.imshow(image)
            # plt.show()

        return perception, neighbors_count
    
    def perform(self, _cells, _perception, _neighbors, _model):
        for i in range(len(_cells)):
            cell = self.cells[_cells[i]]
            perception = _perception[i]
            res = _model(perception).cpu().detach().numpy()
            
            nn_color = res[0:4] * self.color_scale
            nn_hidden = res[4:16]
            nn_move = res[16:18] * self.move_scale
            nn_angle = res[18]
            
            # print (f'nn_color: {nn_color}')
            # print (f'nn_hidden: {nn_hidden}')
            # print (f'nn_move: {nn_move}')
            # print (f'nn_angle: {nn_angle}')
            
            cell.update(nn_color, nn_hidden, nn_move, nn_angle)
            self.grid.update_cell_chunk(cell)
                
        # * perform size-changing actions:
        # new_cells = self.cells.copy()
        # for i in range(len(_cells)):
        #     cell = self.cells[_cells[i]]
        #     neighbors = _neighbors[i]

        #     # * duplicate cell in random direction
        #     if action == 0 and len(new_cells) < self.cell_limit and _neighbors[i] <= self.neighbor_limit:
        #         #print ('duplicate!')
        #         dup_pos = cell.pos + ((np.random.rand(3) * 2.0) - 1) * self.move_scale
        #         dup_pos[2] = 0.0
        #         new_cell = RadialCell(dup_pos, cell.channels, self.id_count)
        #         self.grid.add_cell(new_cell)
        #         new_cells.append(new_cell)
        #         self.id_count += 1
                
        #     # * destroy cell
        #     if self.cells[cell].color.a < 0.1:
        #         old_cell = self.cells[cell]
        #         new_cells.remove(old_cell)
        #         self.grid.remove_cell(old_cell)

        # self.cells = new_cells