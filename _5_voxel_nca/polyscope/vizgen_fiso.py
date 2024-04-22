import polyscope as ps
import numpy as np
from matplotlib import colormaps


_ITER_ = 200
#_PATH_ = f'../../obj/minicube5_isoRmat_v1_thesis/iter_{_ITER_}.npy'
#_PATH_ = f'../../obj/burger_isoRmat_v0_thesis/iter_{_ITER_}.npy'
_PATH_ = f'../../obj/sphere16_isoRmat_v0_regen_thesis/init_{_ITER_}.npy'
# _PATH_ = f'../../obj/sphere16_iso3_v0_thesis/iter_{_ITER_}.npy'
_SCALE_ = 1.0

def euler2rot_np(ax, ay, az):
    r11 = np.cos(ay)*np.cos(az)
    r12 = np.sin(ax)*np.sin(ay)*np.cos(az) - np.sin(az)*np.cos(ax)
    r13 = np.sin(ay)*np.cos(ax)*np.cos(az) + np.sin(ax)*np.sin(az)
    
    r21 = np.sin(az)*np.cos(ay)
    r22 = np.sin(ax)*np.sin(ay)*np.sin(az) + np.cos(ax)*np.cos(az)
    r23 = np.sin(ay)*np.sin(az)*np.cos(ax) - np.sin(ax)*np.cos(az)
    
    r31 = -np.sin(ax)
    r32 = np.sin(ax)*np.cos(ay)
    r33 = np.cos(ax)*np.cos(ay)
    
    R = np.zeros([3, 3], dtype=float)
    
    R[0, 0] = r11
    R[0, 1] = r12
    R[0, 2] = r13
    
    R[1, 0] = r21
    R[1, 1] = r22
    R[1, 2] = r23
    
    R[2, 0] = r31
    R[2, 1] = r32
    R[2, 2] = r33
    
    return R

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
    
    cm = colormaps['twilight']
    
    # * initialize polyscope
    ps.init()
    
    # * create orientation
    angles = []
    colors = []
    for x in range(size):
        for y in range(size):
            for z in range(size):
                
                rx = data[:, 15, x, y, z][0]
                ry = data[:, 14, x, y, z][0]
                rz = data[:, 13, x, y, z][0]
                
                R = euler2rot_np(rx, rz, ry)
                pos = np.matmul(R, np.array([1.0, 0.0, 0.0], dtype=float))

                angles.append(pos)
                colors.append(pos)

    angles = np.array(angles)
    colors = np.array(colors)
    
    # i, j, _ = angles.shape
    # angles = angles.reshape([i, j])
    # colors = colors.reshape([i, j])
    
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
            
            # * show certain vectors
            thresh = 0.5
            
            if c[0] > thresh and c[1] < 0.5 and c[2] < 0.5:
                point_cloud.add_vector_quantity('angle', a, enabled=True, radius=0.2, length=0.05, color=c)
    
    rad = 0.04
    dot = 25
    ps.register_point_cloud(f'black', np.array([0.0, 0.0, 0.0])[None, ...], radius=rad, color=(0, 0, 0))
    ps.register_point_cloud(f'white', np.array([dot, dot, dot])[None, ...], radius=rad, color=(1, 1, 1))
    
    ps.register_point_cloud(f'red', np.array([dot, 0.0, 0.0])[None, ...], radius=rad, color=(1, 0, 0))
    ps.register_point_cloud(f'green', np.array([0.0, dot, 0.0])[None, ...], radius=rad, color=(0, 1, 0))
    ps.register_point_cloud(f'blue', np.array([0.0, 0.0, dot])[None, ...], radius=rad, color=(0, 0, 1))
    
    ps.register_point_cloud(f'yellow', np.array([dot, dot, 0.0])[None, ...], radius=rad, color=(1, 1, 0))
    ps.register_point_cloud(f'magenta', np.array([dot, 0.0, dot])[None, ...], radius=rad, color=(1, 0, 1))
    ps.register_point_cloud(f'cyan', np.array([0.0, dot, dot])[None, ...], radius=rad, color=(0, 1, 1))

    # View the point cloud and mesh we just registered in the 3D UI
    ps.show()

if __name__ == '__main__':
    main()