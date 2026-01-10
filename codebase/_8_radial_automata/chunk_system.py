from utility import *

class ChunkSystem:
    def __init__(self, _size):
        self.size = _size
        self.chunks = {}
        
    def print_state(self):
        print (f'chunk_system_state:\n{self.chunks}')

    def add_cell(self, _cell):
        c = self.pos_to_chunk(_cell.pos)
        _cell.chunk = c
        k = str(c)
        if k not in self.chunks:
            self.chunks[k] = []
        self.chunks[k].append(_cell)

    def remove_cell(self, _cell):
        c = _cell.chunk
        k = str(c)
        self.chunks[k].remove(_cell)

    def update_cell_chunk(self, _cell):
        c = _cell.chunk
        nc = self.pos_to_chunk(_cell.pos)
        if (c != nc).all():
            self.chunks[str(c)].remove(_cell)
            self.add_cell(_cell)

    def pos_to_chunk(self, _pos):
        return np.ceil(_pos.xy() / self.size).astype(int)

    def query_neighbors_in_radius(self, _cell):
        neighbors = []
        chunk = _cell.chunk
        for x in range(int(chunk[0])-1, int(chunk[0])+2, 1):
            for y in range(int(chunk[1])-1, int(chunk[1])+2, 1):
                k = str(np.array([x, y]).astype(int))
                if k in self.chunks:
                    for cell in self.chunks[k]:
                        if cell.id != _cell.id:
                            dis = math.dist(_cell.pos.xy(), cell.pos.xy())
                            if dis < self.size:
                                neighbors.append(cell)
        return neighbors