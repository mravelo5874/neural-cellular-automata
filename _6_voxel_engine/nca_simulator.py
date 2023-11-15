import sys
import os

dir = os.getcwd().split('\\')[:-1] 
mod = '/'.join(dir)+'/_5_voxel_nca'
sys.path.insert(1, mod)

from scripts.nca.VoxelNCA import VoxelNCA as NCA

class NCASimulator:
    def __init__(self, _model):
        pass