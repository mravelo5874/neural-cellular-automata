import glm

class Utils:
    def ray_cube_intersection(_pos, _vec, _size, _cube):
        # * normalize values
        pos = glm.vec3(_pos)
        vec = glm.normalize(_vec)
        inv = 1/vec
        
        # * get correct cube pos
        cube = (_cube/_size*2)-1
        unit = 2/_size

        min_bb = glm.vec3(cube-unit)
        max_bb = glm.vec3(cube+unit)

        t_min_x = (min_bb.x - pos.x) * inv.x
        t_max_x = (max_bb.x - pos.x) * inv.x
        t_min_y = (min_bb.y - pos.y) * inv.y
        t_max_y = (max_bb.y - pos.y) * inv.y
        t_min_z = (min_bb.z - pos.z) * inv.z
        t_max_z = (max_bb.z - pos.z) * inv.z

        # * find the minimum and maximum t-values for the entry and exit points
        t_min = max(min(t_min_x, t_max_x), min(t_min_y, t_max_y), min(t_min_z, t_max_z))
        t_max = min(max(t_min_x, t_max_x), max(t_min_y, t_max_y), max(t_min_z, t_max_z))

        # * if minimum t-value for the exit point is greater than the maximum t-value for the entry point, there is no intersection
        if t_min > t_max:
            return -1

        # * get absolute val of t_min
        return abs(t_min)

