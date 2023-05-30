from pathlib import Path
import numpy as np
import pypcd


def get_scan_from_pcloud(pcloud):
    ''' 从pypcd的数据结构中获取点云数据，生成numpy.array格式
    pcloud: pypcd.PointCloud
    '''
    scan = np.empty((pcloud.points, 4), dtype=np.float32)
    scan[:, 0] = pcloud.pc_data['x']
    scan[:, 1] = pcloud.pc_data['y']
    scan[:, 2] = pcloud.pc_data['z']
    try:
        scan[:, 3] = pcloud.pc_data['intensity']
    except ValueError:
        scan[:, 3] = 255.0
    return scan


def load_pcd(f_pcd):
    ''' 从pcd文件读取点云数据，将数据到数据结构pypcd.PointCloud
    f_pcd: strin, pcd文件路径
    '''
    try:
        if isinstance(f_pcd, str) or isinstance(f_pcd, Path):
            pcloud = pypcd.PointCloud.from_path(f_pcd)
        else:
            raise TypeError(f'load_pcd do not support type {type(f_pcd)}')

    except AssertionError:
        print ("Assertion when load pcd: %s" % f_pcd)
        return None
    scan = get_scan_from_pcloud(pcloud)
    scan[:, 3] /= 255.0
    return scan
