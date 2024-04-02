import polyscope as ps
import numpy as np
from matplotlib import colormaps

_ITER_ = 200
#_PATH_ = f'../../obj/rubiks_black_cube_iso3_v3_4/iter_{_ITER_}.npy'
_PATH_ = f'../../obj/gray_cactus_iso3_v0_0/iter_{_ITER_}.npy'
_SCALE_ = 1.0

def main():
    
    data = np.load(_PATH_)
    print (f'data.shape: {data.shape}')
    size = data.shape[-1]
    
    # * create points array for polyscope
    points = []
    for x in range(size):
        for y in range(size):
            for z in range(size):
                points.append(np.array([x, y, z]))
    points = np.array(points)
    print (f'points.shape: {points.shape}')
    
    # * initialize polyscope
    ps.init()

    cm = colormaps['hsv']
    
    # * create orientation
    angles = []
    colors = []
    for x in range(size):
        for y in range(size):
            for z in range(size):
                
                a = data[:, 15, x, y, z]
                c = np.cos(a)[0]
                s = np.sin(a)[0]
                # print (f'angle for {x}, {y}, {z}: {a} rad -> cos(a): {c}, sin(a): {s}')
                if data[:, 3, x, y, z] > 0.1:
                    angles.append(np.array([c, 0.0, s]))
                    n = a / (2*np.pi)
                    w = cm(n)[0, 0:3]
                    colors.append(w)
                else:
                    angles.append(np.array([0.0, 0.0, 0.0]))
                    n = a / (2*np.pi)
                    w = cm(n)[0, 0:3]
                    colors.append(w)
    angles = np.array(angles)
    colors = np.array(colors)
    
    print (f'angles.shape: {angles.shape}')
    print (f'colors.shape: {colors.shape}')
    
    for i in range(len(points)):
        
        x, y, z = points[i]
        if data[:, 3, x, y, z] > 0.1:
            # print (f'x: {x}, y: {y}, z: {z}')
            point = np.array([x, z, y], dtype=float) * _SCALE_
            point = point[None, ...]
            
            
            point_cloud = ps.register_point_cloud(f'{x}-{y}-{z}', point, radius=0)
            
            a = angles[i][None, ...]
            c = tuple(colors[i])
            # print (f'angle: {a}')
            # print (f'color: {c}')
            
            point_cloud.add_vector_quantity('angle', a, enabled=True, radius=0.1, color=c)
    
    # ### Register a mesh
    # # `verts` is a Nx3 numpy array of vertex positions
    # # `faces` is a Fx3 array of indices, or a nested list
    # ps.register_surface_mesh("my mesh", verts, faces, smooth_shade=True)

    # # Add a scalar function and a vector function defined on the mesh
    # # vertex_scalar is a length V numpy array of values
    # # face_vectors is an Fx3 array of vectors per face
    # ps.get_surface_mesh("my mesh").add_scalar_quantity("my_scalar", 
    #         vertex_scalar, defined_on='vertices', cmap='blues')
    # ps.get_surface_mesh("my mesh").add_vector_quantity("my_vector", 
    #         face_vectors, defined_on='faces', color=(0.2, 0.5, 0.5))

    # View the point cloud and mesh we just registered in the 3D UI
    ps.show()

if __name__ == '__main__':
    main()