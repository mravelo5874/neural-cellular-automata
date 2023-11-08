import glm
import math
import numpy as np
from numba import njit

# * resolution
WIN_RES = glm.vec2(1600, 900)

# * camera
ASPECT_RATIO = WIN_RES.x/WIN_RES.y
FOV_DEG = 50
VERT_FOV = glm.radians(FOV_DEG)
HORZ_FOV = math.atan(math.tan(VERT_FOV*0.5)*ASPECT_RATIO)
NEAR = 0.1
FAR = 2000.0
PITCH_MAX = glm.radians(89)

# * player
PLAYER_SPEED = 0.005
PLAYER_ROT_SPEED = 0.003
PLAYER_INIT_POS = glm.vec3(0, 0, 1)
MOUSE_SENS = 0.002

# * colors
BG_COLOR = glm.vec3(0.1, 0.16, 0.25)