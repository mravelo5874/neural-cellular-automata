import glm

class Utils:
    def is_point_in_box(_point, _minbb, _maxbb):
        return (_minbb[0] <= _point[0] <= _maxbb[0] and 
                _minbb[1] <= _point[1] <= _maxbb[1] and 
                _minbb[2] <= _point[2] <= _maxbb[2])
    
    def ray_box_intersection(_pos, _vec, _minbb, _maxbb):
        # * normalize values
        pos = glm.vec3(_pos)
        vec = glm.normalize(_vec)
        inv = 1/vec
        
        min_bb = _minbb
        max_bb = _maxbb

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
            return None, None

        # * get absolute val of t_min
        return t_min, t_max

    def ray_plane_intersection(_pos, _vec, _pln_p, _pln_norm):
        # Calculate the dot product of the plane normal and ray direction
        dot = glm.dot(_vec, _pos)

        # If the dot product is 0, the ray is parallel to the plane and there is no intersection
        if dot == 0:
            return None

        # Calculate the scalar t
        t = glm.dot(_pln_norm, (_pln_p-_pos))/dot

        # If t is negative, the intersection point is behind the ray's origin
        if t < 0:
            return None
        
        return t