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
    affine_inv: np.array(3, 4), 仿射变换的逆变换矩阵
    """
    rotation = affine_matrix[:3, :3]
    translation = affine_matrix[:3, 3]
    rotation_inv = np.linalg.inv(rotation)
    translation_inv = - np.dot(rotation_inv, translation[:, np.newaxis])

    affine_inv = np.concatenate((rotation_inv, translation_inv), axis=1)
    return affine_inv


def trans_pts_by_affine(pts, affine_matrix):
    """ 根据仿射函数对点进行转换
    pts: np.array(N, 3), 待变换的点，N表示点的个数
    affine_matrix: np.array(3, 4), [R, T], 3*4的矩阵，R为旋转矩阵，T为平移矩阵
    """
    rotation_matrix = affine_matrix[:3, :3]
    translation = affine_matrix[:3, 3]

    trans_points = np.dot(rotation_matrix, pts.T) + translation[:, np.newaxis]
    trans_points = trans_points.T # (3, N) -> (N, 3)
    return trans_points


def trans_heading_by_rotation(heading, rotation):
    """ 将heading角根据旋转矩阵从一个坐标系转到另一个坐标系
    heading:
    rotation
    """
    rotation = Rotation.from_matrix(rotation)
    yaw = np.ones_like(heading) * rotation.as_euler('zyx')[0]
    heading = heading + yaw
    return heading


def matrix_to_string(matrix):
    ss = list(matrix.reshape(-1))
    ss = list(map(str, ss))
    return ' '.join(ss)


def box3d_pts_2d_to_bbox_2d(box3d_pts_2d):
    """ 将2d图像上的3d框转化到2dbbox
    box3d_pts_2d: np.array(N, 8, 2) 在图像上障碍物的N个框，每个框8个角点，
    bbox_2d: np.array(N, 4) 在图像上N个2d框[x_min, y_min, x_max, y_max]
    """
    # 提取立体框的每个顶点的坐标
    x = box3d_pts_2d[:, :, 0]
    y = box3d_pts_2d[:, :, 1]

    # 计算2D边界框的坐标
    x_min = np.min(x, axis=1)
    y_min = np.min(y, axis=1)
    x_max = np.max(x, axis=1)
    y_max = np.max(y, axis=1)

    # 组合2D边界框的坐标为数组
    bbox_2d = np.stack((x_min, y_min, x_max, y_max), axis=1)

    return bbox_2d


class AicvCalibration(object):
    """ aicv数据集中的标注参数 
    cam_K: np.array(3,3) 相机内参矩阵
    cam2lidar: np.array(3,4) 相机坐标系到lidar坐标系的变换矩阵
    lidar2cam: np.array(3,4) lidar坐标系到相机坐标系的变换矩阵
    """
    def __init__(self, aicv_calib_file_path, cam_name, scale=None):
        self.aicv_cam_name = cam_name
        self._parse_aicv_calib_file(aicv_calib_file_path)
        if scale is not None:
            self.cam_K[:2, :] = scale * self.cam_K[:2, :]
    
    def read_aicv_calib_file(self, aicv_calib_file_path):
        """ 读取aicv数据的param.txt文件 """
        with open(aicv_calib_file_path, 'r') as param_txt:
            param = param_txt.read().splitlines()
        return param
    
    def _parse_cam_K(self, param):
        """ 解析aicv的params.txt文件, 获取相机内参矩阵cam_K 
        cam_K: np.array(3,3) 相机内参矩阵
        """
        intrinsic_key = self.aicv_cam_name + '_K'
        for line in param:
            if intrinsic_key in line:
                intrinsic_line = line
                break
        intrinsic_line = intrinsic_line.split(' ')[1:]
        intrinsic_line = list(map(float, intrinsic_line))
        cam_K = np.array(intrinsic_line).reshape(3,3)
        return cam_K

    def _parse_cam2lidar(self, param):
        """ 解析aicv的params.txt文件, 获取相机外参矩阵cam2lidar

        """
        extrinsic_key = 'calib_' + self.aicv_cam_name + '_to_at128_fusion'
        for line in param:
            if extrinsic_key in line:
                extrinsic_line = line
                break
        extrinsic_line = extrinsic_line.split(' ')
        translation = [float(e) for e in extrinsic_line[1:4]]
        quaternion = [float(e) for e in extrinsic_line[4:]]

        rotation_matrix = quaternion_to_rotation_matrix(quaternion)
        translation_matrix = np.array(translation)[:, np.newaxis]

        cam2lidar = np.concatenate((rotation_matrix, translation_matrix), axis=1)
        return cam2lidar

    def _parse_aicv_calib_file(self, aicv_calib_file_path):
        param = self.read_aicv_calib_file(aicv_calib_file_path)
        self.cam_K = self._parse_cam_K(param)
        self.cam2lidar = self._parse_cam2lidar(param)
        self.lidar2cam = get_affine_matrix_inv(self.cam2lidar)

    def write_to_kitti_calib_file(self, kitti_calib_file_path):
        """ 将参数写成kitti格式的文件 """
        kitti_calib_param = [
            'P0: 1.0 0.0 0.0 0.0 1.0 0.0 0.0 0.0 1.0 0.0 0.0 0.0',
            'P1: 1.0 0.0 0.0 0.0 1.0 0.0 0.0 0.0 1.0 0.0 0.0 0.0',
            'R_rect 1.0 0.0 0.0 0.0 1.0 0.0 0.0 0.0 1.0',
            'Tr_imu_velo 1.0 0.0 0.0 0.0 0.0 1.0 0.0 0.0 0.0 0.0 1.0 0.0'
            ]
        # 把内参矩阵P2写入kitti-calib格式文件
        kitti_cam_intrinsic = np.concatenate((self.cam_K, np.zeros((3, 1))), axis=1)
        kitti_cam_intrinsic_string = matrix_to_string(kitti_cam_intrinsic)
        kitti_cam_intrinsic_string = 'P2: ' + kitti_cam_intrinsic_string
        kitti_calib_param.insert(2, kitti_cam_intrinsic_string)

        # 把内参矩阵P3写入kitti-calib格式文件，与P2一致
        kitti_cam_intrinsic_string = kitti_cam_intrinsic_string.replace('P2:', 'P3:')
        kitti_calib_param.insert(3, kitti_cam_intrinsic_string)

        # 把Tr_velo_cam写入kitti-calib格式文件
        kitti_tr_velo_cam_string = matrix_to_string(self.lidar2cam)
        kitti_tr_velo_cam_string = 'Tr_velo_cam ' + kitti_tr_velo_cam_string
        kitti_calib_param.insert(5, kitti_tr_velo_cam_string)

        with open(kitti_calib_file_path, 'w') as f:
            for line in kitti_calib_param:
                f.write(line + '\n')

    def trans_heading_from_aicv_to_kitti(self, heading):
        """ 将heading角根据旋转矩阵从一个坐标系转到另一个坐标系
        eg.
            heading: lidar坐标系下的航向角
            lidar2cam: lidar坐标系到cam坐标系的变换矩阵
        """
        rotation = Rotation.from_matrix(self.lidar2cam[:3, :3])
        yaw = np.ones_like(heading) * rotation.as_euler('zyx')[0]
        heading = heading + yaw
        return heading

    # ===========================
    # ------- 3d to 3d ----------
    # ===========================
    def project_lidar_to_cam(self, pts_lidar):
        pts_cam = trans_pts_by_affine(pts_lidar, self.lidar2cam)
        return pts_cam

    def project_cam_to_lidar(self, pts_cam):
        pts_lidar = trans_pts_by_affine(pts_cam, self.cam2lidar)
        return pts_lidar

    # ===========================
    # ------- 3d to 2d ----------
    # ===========================
    def project_lidar_to_pixel(self, pts_lidar):
        pass

        
        

