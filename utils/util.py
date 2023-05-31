from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
import pypcd

PI = 3.14152653589793


def reg_radian(radian):
    '''值域规则化到(-pi, pi], angle为弧度, 输出也为弧度'''
    if radian > PI:
        radian = radian - 2 * PI
        return reg_radian(radian)
    elif radian <= - PI:
        radian = radian + 2 * PI
        return reg_radian(radian)
    else:
        return radian


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


def quaternion_to_rotation_matrix(quaternion):
    # 创建旋转对象
    rotation = Rotation.from_quat(quaternion)
    # 获取旋转矩阵
    rotation_matrix = rotation.as_matrix()

    return rotation_matrix


def trans_cam_to_pixel(cam_points, calib_cam_K):
    """ 将相机坐标系下的点转化为像素坐标系
    camera_points: shape(N, 3), 相机坐标系下的点
    calib_cam_K: list(9), 相机内参矩阵， [fx, 0, cx, 0, fy, cy, 0, 0, 1]
    points_pixel: shape(N, 2) 像素坐标系下的点
    """
    calib_cam_K = np.array(calib_cam_K).reshape((3, 3))
    # 将相机坐标系下的点转换为像素坐标系
    points_pixel = np.dot(calib_cam_K, cam_points.T).T

    # 将齐次坐标转换为二维坐标
    points_pixel[:, 0] /= points_pixel[:, 2]
    points_pixel[:, 1] /= points_pixel[:, 2]
    points_pixel = points_pixel[:, :2]

    return points_pixel


def trans_lidar_to_cam(lidar_points, calib_cam2lidar):
    """ 将lidar坐标系下的点转化为相机坐标系下的点
    lidar_points: np.array, shape(N, 3), 雷达坐标系下的点
    calib_cam2lidar: list(7), [Tx, Ty, Tz, Rx, Ry, Rz, Rw] 相机坐标系到lidar坐标系的转化pose
    return:
        cam_points: np.array(N, 3), 相机坐标系下的点
    """
    # 解析转换信息
    Tx, Ty, Tz, Rx, Ry, Rz, Rw = calib_cam2lidar

    # 构建旋转矩阵
    rotation_matrix = quaternion_to_rotation_matrix([Rx, Ry, Rz, Rw])
    rotation_matrix_inv = np.linalg.inv(rotation_matrix) # lidar point to cam

    # 构建平移向量
    translation_vector = np.array([Tx, Ty, Tz]) # cam point to lidar

    # 将LiDAR坐标系转化为相机坐标系下的点
    cam_points = np.dot(rotation_matrix_inv, lidar_points.T - translation_vector[:, np.newaxis])
    cam_points = cam_points.T # (3, N) -> (N, 3)
    return cam_points


def trans_lidar_to_pixel(lidar_points, calib_cam2lidar, calib_cam_K):
    """ 将lidar坐标系下的点转化为像素坐标系下的点
    lidar_points: np.array, shape(N, 3), 雷达坐标系下的点
    calib_cam2lidar: list(7), [Tx, Ty, Tz, Rx, Ry, Rz, Rw] 相机坐标系到lidar坐标系的转化pose
    calib_cam_K: list(9), 相机内参矩阵， [fx, 0, cx, 0, fy, cy, 0, 0, 1]
    """
    # 将lidar坐标系下的点转化为相机坐标系下的点
    cam_points = trans_lidar_to_cam(lidar_points, calib_cam2lidar)

    # 将相机坐标系下的点转化为像素坐标系下的点
    pixel_points = trans_cam_to_pixel(cam_points, calib_cam_K)

    return pixel_points


def get_affine_matrix_inv(affine_matrix):
    """ 从一个仿射矩阵获得其逆变换的矩阵
    affine_matrix: np.array(3, 4), [R, T], 3*4的矩阵，R为旋转矩阵，T为平移矩阵

    """
    rotation = affine_matrix[:3, :3]
    translation = affine_matrix[:3, 3]
    rotation_inv = np.linalg.inv(rotation)
    translation_inv = - np.dot(rotation_inv, translation[:, np.newaxis])

    affine_inv = np.concatenate((rotation_inv, translation_inv), axis=1)
    return affine_inv


class AicvCalibration(object):
    """ aicv数据集中的标注参数 """
    def __init__(self, aicv_calib_file_path):
        self.parse_aicv_calib_file(aicv_calib_file_path)
        self.cam2lidar = 0
        self.lidar2cam = 0
        self.cam_K = 0
        pass

    def project_lidar_to_cam(self, pts_lidar):
        pts_cam = trans_lidar_to_cam(pts_lidar, self.lidar2cam)
        return pts_cam
    
    def parse_aicv_calib_file(aicv_calib_file_path):
        pass
    
    def write_to_kitti_calib_file(self, kitti_calib_file_path):
        pass
    