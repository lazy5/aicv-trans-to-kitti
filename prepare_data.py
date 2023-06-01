
""" 根据从aicv下载的数据转化为kitti-mot格式的数据

Author: fangchenyu
Date: 2023.5.30
"""
from ast import Num
import os
import tempfile
import zipfile
import copy
import json
import shutil
import concurrent.futures as futures

import pandas as pd
import numpy as np

from utils.util import load_pcd, reg_radian, AicvCalibration, box3d_pts_2d_to_bbox_2d
from utils.kitti_util import boxes_to_corners_3d


def mkdir(path):
    file_dir = os.path.dirname(path)
    if not os.path.exists(file_dir):
        os.makedirs(file_dir)


def extract_zip_to_temp(zip_file_path):
    # 创建临时文件夹
    temp_dir = tempfile.TemporaryDirectory()

    # 解压缩zip文件到临时文件夹
    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        zip_ref.extractall(temp_dir.name)

    # 返回临时文件夹的路径
    return temp_dir


def write_line(f, frame_idx, track_id_i, type_i, truncated_i, occluded_i, alpha_i, 
               bbox_i, dimensions_i, location_i, rotation_y_i, score_i):
    line = f'{int(frame_idx)} {int(track_id_i)} {type_i} {truncated_i} {occluded_i} {alpha_i} \
{bbox_i[0]} {bbox_i[1]} {bbox_i[2]} {bbox_i[3]} \
{dimensions_i[0]} {dimensions_i[1]} {dimensions_i[2]} \
{location_i[0]} {location_i[1]} {location_i[2]} \
{rotation_y_i} {score_i}'
    f.write(line + '\n')


def trans_lidar_file(file_path, aicv_infos_dict, idx, kitti_path):
    kitti_lidar_file_path = os.path.join(kitti_path, 'velodyne/0001', "%06d.bin" % (idx))
    pc_data = load_pcd(file_path)
    mkdir(kitti_lidar_file_path)
    pc_data.tofile(kitti_lidar_file_path)
    aicv_infos_dict[idx]['lidar_file_path'] = kitti_lidar_file_path


def trans_img_file(file_path, aicv_infos_dict, idx, kitti_path):
    kitti_img_file_path = os.path.join(kitti_path, 'image_02/0001', "%06d.png" % (idx))
    mkdir(kitti_img_file_path)
    shutil.move(file_path, kitti_img_file_path)
    aicv_infos_dict[idx]['image_file_path'] = kitti_img_file_path


def trans_label_file(aicv_infos_dict, sample_idx, kitti_path):
    """ 将label数据进行转化，从lidar坐标系转化为cam坐标系 
    aicv_infos_dict: list(n_frame), 其中每个字段存储n个数据，n表示当前帧的障碍物数量
        frame_id
        trackId: 跟踪ID，np.array(N)，N表示障碍物个数
        gt_names: 障碍物类型，np.array(N)
        gt_boxes: np.array(N, 7), [x, y, z, l, w, h, heading]，lidar坐标系的标注结果，坐标系为前左上
        zip_file_path
        lidar_file_path
        image_file_path
    """
    kitti_label_file_path = os.path.join(kitti_path, 'label_02/0001.txt')
    mkdir(kitti_label_file_path)

    # 获取该段场景的calibration
    aicv_calibration = aicv_infos_dict[sample_idx]['calibration']

    with open(kitti_label_file_path, 'w') as f_kitti_label_file:
        for frame_idx, aicv_info in enumerate(aicv_infos_dict): # frame_idx表示帧序号
            # 对于aicv数据中的location，将lidar坐标系下的标注转化为camera坐标系
            aicv_location = aicv_info['gt_boxes'][:, :3]
            kitti_location = aicv_calibration.project_lidar_to_cam(aicv_location)

            for i in range(len(aicv_info['trackId'])): # i表示当前帧的第i个障碍物
                # 获取在camera坐标系下的bbox
                track_id_i = aicv_info['trackId'][i]
                type_i = aicv_info['gt_names'][i]
                truncated_i = 0 # tips: 表示了目标检测的截断情况，此处设置为0，全部不截断，该参数对nerf应该没有影响
                occluded_i = 0 #tips: 表示该障碍物是否可见，0全部可见，1部分遮挡，2大部分遮挡，3不清楚
                alpha_i = 0 # TODO: 该值表示障碍物的观测角，对nerf无用
                bbox_i = [0, 0, 0, 0]
                dimensions_i = aicv_info['gt_boxes'][i, 3:6]
                location_i = kitti_location[i]
                rotation_y_i = reg_radian(aicv_info['gt_boxes'][i, 6])
                score_i = 1
                write_line(f_kitti_label_file, frame_idx, track_id_i, type_i, truncated_i, occluded_i, alpha_i, 
                        bbox_i, dimensions_i, location_i, rotation_y_i, score_i)
    aicv_infos_dict[sample_idx]['label_file_path'] = kitti_label_file_path

def trans_calib_file(calib_file_path, aicv_infos_dict, sample_idx, kitti_path):
    kitti_calib_file_path = os.path.join(kitti_path, 'calib/0001.txt')
    mkdir(kitti_calib_file_path)

    # 解析aicv文件，输出一个kitti-mot格式文件
    aicv_calib = AicvCalibration(calib_file_path, 'obstacle')
    aicv_calib.write_to_kitti_calib_file(kitti_calib_file_path)
    aicv_infos_dict[sample_idx]['calibration'] = aicv_calib


def trans_oxts_file(pose_file_path, aicv_infos_dict, sample_idx, kitti_path):
    with open(pose_file_path, 'r') as aicv_file:
        line = aicv_file.read().strip().replace('\t', ' ')
    aicv_infos_dict[sample_idx]['pose'] = line


def parse_aicv_anno(anno_infos, root_path, num_workers=16):
    def get_single_info(sample_idx):
        info = copy.deepcopy(anno_infos.iloc[sample_idx, :])
        sample_idx = info['_id']
        result = json.loads(info['result'])
        zip_file_path = result['datasetsRelatedFiles'][1]['bos_key'].replace('records/kitti_data/at128_fusion', root_path)
        num_obj = result['labelData']['markData']['numberCube3d']
        annos = result['labelData']['markData']['cube3d']
        loc, dims, rots, gt_names, trackId = [], [], [], [], []
        for i in range(num_obj):
            loc.append([annos[i]['position']['x'], annos[i]['position']['y'], annos[i]['position']['z']])
            dims.append(annos[i]['size'])
            rots.append(annos[i]['rotation']['phi'])
            gt_names.append(annos[i]['type'])
            trackId.append(annos[i]['trackId'])
        loc, dims, rots, gt_names, trackId = np.array(loc), np.array(dims), np.array(rots), np.array(gt_names), np.array(trackId)
        gt_boxes_lidar = np.concatenate([loc, dims, rots[..., np.newaxis]], axis=1).astype(np.float32) # [x, y, z, l, w, h, r_z]
        # gt_boxes_camera = box_utils.boxes3d_aicv_to_kitti_camera(gt_boxes_lidar) # [x, y, z, l, h, w, r]
        # annos_camera_coordinate = box_utils.boxes3d_kitti_camera_to_annos(gt_boxes_camera, gt_names, gt_boxes_lidar)
        
        info_dict = {
            'frame_id': sample_idx,
            'trackId': trackId,
            'gt_names': gt_names,
            'gt_boxes': gt_boxes_lidar,
            'zip_file_path': zip_file_path
        }
        # pcd_file = root_split_path / pcd_path
        # assert pcd_file.exists()
        return info_dict
    
    sample_id_list = list(range(len(anno_infos)))
    with futures.ThreadPoolExecutor(num_workers) as executor:
        infos = executor.map(get_single_info, sample_id_list)

    return list(infos)


def process_temp_dir(aicv_infos_dict, kitti_path, num_workers=16):
    def process_single_frame(sample_idx):
        zip_file_path = aicv_infos_dict[sample_idx]['zip_file_path']
        temp_dir = extract_zip_to_temp(zip_file_path)
        # print(os.listdir(temp_dir.name))
        # print(os.listdir(os.path.join(temp_dir.name, 'velodyne_points')))

        # 处理临时文件夹中的文件
        pcd_file_path = os.path.join(temp_dir.name, 'velodyne_points/at128_fusion.pcd')
        trans_lidar_file(pcd_file_path, aicv_infos_dict, sample_idx, kitti_path) # 处理lidar数据
        img_file_path = os.path.join(temp_dir.name, 'images/obstacle/image.jpg')
        trans_img_file(img_file_path, aicv_infos_dict, sample_idx, kitti_path) # 处理图像数据
        pose_file_path = os.path.join(temp_dir.name, 'velodyne_points/pose.txt')
        trans_oxts_file(pose_file_path, aicv_infos_dict, sample_idx, kitti_path)
        if sample_idx == 0: # 对于标定文件和label文件，将输出单个整合文件
            calib_file_path = os.path.join(temp_dir.name, 'params/params.txt')
            trans_calib_file(calib_file_path, aicv_infos_dict, sample_idx, kitti_path) # 处理相机和激光雷达内外参数据
            trans_label_file(aicv_infos_dict, sample_idx, kitti_path) # 处理label数据

        temp_dir.cleanup()
        
    # process_single_frame(0) # debug: 测试单帧数据处理过程
    sample_id_list = list(range(len(aicv_infos_dict)))
    with futures.ThreadPoolExecutor(num_workers) as executor:
        executor.map(process_single_frame, sample_id_list)
        # infos = executor.map(process_single_frame, sample_id_list)

    # 将pose信息写入文件
    kitti_oxts_file_path = os.path.join(kitti_path, 'oxts/0001.txt')
    mkdir(kitti_oxts_file_path)
    with open(kitti_oxts_file_path, 'w') as kitti_file:
        for i, aicv_info in enumerate(aicv_infos_dict):
            pose_line = aicv_infos_dict[i]['pose']
            kitti_file.write(pose_line + '\n')


def tran_aicv_to_kitti_pipline(root_path, kitti_path):
    """ 将原始aicv数据转化为kitti-mot格式数据，仅支持处理一个场景
    root_path: 存放一个场景数据的根目录
    """
    # 解析result.txt文件
    label_file_aicv = os.path.join(root_path, 'result.txt')
    aicv_infos = pd.read_csv(label_file_aicv, sep='\t')
    aicv_infos_dict = parse_aicv_anno(aicv_infos, root_path)
    print(len(aicv_infos_dict))

    process_temp_dir(aicv_infos_dict, kitti_path)


if __name__ == '__main__':
    root_path = 'data/dataid-sample'
    kitti_path = 'data/aicv-kitti'
    tran_aicv_to_kitti_pipline(root_path, kitti_path)
