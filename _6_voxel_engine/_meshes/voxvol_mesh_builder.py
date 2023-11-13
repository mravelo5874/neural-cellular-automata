from settings import *

def is_void(_voxel_pos, _chunk_voxels):
    x, y, z = _voxel_pos
    if 0 <= x < CHUNK_SIZE and 0 <= y < CHUNK_SIZE and 0 <= z < CHUNK_SIZE:
        if _chunk_voxels[x+CHUNK_SIZE*z+CHUNK_AREA*y]:
            return False
    return True

def add_data(_vertex_data, _index, *vertices):
    for vertex in vertices:
        for attr in vertex:
            _vertex_data[_index] = attr
            _index += 1
    return _index

def build_voxvol_mesh(_chunk_voxels, _format_size):
    # * CHUNK_VOL = number of voxels in chunk
    # * 18 = max 3 faces per voxel = 6 triangles = 18 vertices
    # * 5 = 5 attributes per vertex: x, y, z, voxel_id, face_id
    # * format: [x, y, z, voxel_id, face_id]
    vertex_data = np.empty(CHUNK_VOL*18*_format_size, dtype='uint8')
    index = 0
    
    for x in range(CHUNK_SIZE):
        for y in range(CHUNK_SIZE):
            for z in range(CHUNK_SIZE):
                voxel_id = _chunk_voxels[x+CHUNK_SIZE*z+CHUNK_AREA*y]
                if not voxel_id:
                    continue
                
                # * top face
                if is_void((x, y+1, z), _chunk_voxels):
                    v0 = (x  , y+1, z  , voxel_id, 0)
                    v1 = (x+1, y+1, z  , voxel_id, 0)
                    v2 = (x+1, y+1, z+1, voxel_id, 0)
                    v3 = (x  , y+1, z+1, voxel_id, 0)
                    index = add_data(vertex_data, index, v0, v3, v2, v0, v2, v1)
                    
                # * bottom face
                if is_void((x, y-1, z), _chunk_voxels):
                    v0 = (x  , y  , z  , voxel_id, 1)
                    v1 = (x+1, y  , z  , voxel_id, 1)
                    v2 = (x+1, y  , z+1, voxel_id, 1)
                    v3 = (x  , y  , z+1, voxel_id, 1)
                    index = add_data(vertex_data, index, v0, v2, v3, v0, v1, v2)
                    
                # * right face
                if is_void((x+1, y, z), _chunk_voxels):
                    v0 = (x+1, y  , z  , voxel_id, 2)
                    v1 = (x+1, y+1, z  , voxel_id, 2)
                    v2 = (x+1, y+1, z+1, voxel_id, 2)
                    v3 = (x+1, y  , z+1, voxel_id, 2)
                    index = add_data(vertex_data, index, v0, v1, v2, v0, v2, v3)
                    
                # * left face
                if is_void((x-1, y, z), _chunk_voxels):
                    v0 = (x  , y  , z  , voxel_id, 3)
                    v1 = (x  , y+1, z  , voxel_id, 3)
                    v2 = (x  , y+1, z+1, voxel_id, 3)
                    v3 = (x  , y  , z+1, voxel_id, 3)
                    index = add_data(vertex_data, index, v0, v2, v1, v0, v3, v2)
                    
                # * back face
                if is_void((x, y, z-1), _chunk_voxels):
                    v0 = (x  , y  , z  , voxel_id, 4)
                    v1 = (x  , y+1, z  , voxel_id, 4)
                    v2 = (x+1, y+1, z  , voxel_id, 4)
                    v3 = (x+1, y  , z  , voxel_id, 4)
                    index = add_data(vertex_data, index, v0, v1, v2, v0, v2, v3)
                    
                # * front face
                if is_void((x, y, z+1), _chunk_voxels):
                    v0 = (x  , y  , z+1, voxel_id, 5)
                    v1 = (x  , y+1, z+1, voxel_id, 5)
                    v2 = (x+1, y+1, z+1, voxel_id, 5)
                    v3 = (x+1, y  , z+1, voxel_id, 5)
                    index = add_data(vertex_data, index, v0, v2, v1, v0, v3, v2)
                    
    return vertex_data[:index+1]