# usage:
#       bash scripts/vrl3_gen_demonstration_expert.sh door
import argparse
import os
import torch
from termcolor import cprint
import zarr
from copy import deepcopy
import numpy as np
import yaml

from pathlib import Path

import h5py
import cv2
import open3d as o3d
import pytorch3d.ops as torch3d_ops
import pytorch3d.transforms as torch3d_tf

import visualizer

# constant
from constant import *
# FX, FY, CX, CY = 72.70, 72.72, 42.0, 42.0

# Default CROP_BOUNDS if not provided by config
# CROP_BOUNDS = [-0.8, 0.8, 0.3, 1.3, 0.001, 1.0]

def quat_to_rot6d(quat: np.ndarray) -> np.ndarray:
    """Convert quaternion (w, x, y, z) to 6D rotation representation."""
    quat_t = torch.tensor(quat, dtype=torch.float32).view(-1, 4)
    rot6d = torch3d_tf.matrix_to_rotation_6d(torch3d_tf.quaternion_to_matrix(quat_t))
    return rot6d.detach().cpu().numpy().reshape(-1)

def read_hdf5_example(filename='example.h5'):
    """读取并探索HDF5文件内容"""
    
    with h5py.File(filename, 'r') as f:
        '''
        print("=" * 50)
        print(f"文件: {filename}")
        print("=" * 50)
        
        # 1. 显示文件结构
        def print_structure(name, obj):
            indent = '  ' * name.count('/')
            if isinstance(obj, h5py.Dataset):
                print(f"{indent}📊 Dataset: {name.split('/')[-1]} | Shape: {obj.shape} | Dtype: {obj.dtype}")
            elif isinstance(obj, h5py.Group):
                print(f"{indent}📁 Group: {name.split('/')[-1]}")
        
        print("\n📂 文件结构:")
        f.visititems(print_structure)
        '''

        data = {}

        data['ee_control'] = f['ee_control'][:]
        data['ee_states'] = f['ee_states'][:]
        data['gripper_control'] = f['gripper_control'][:]
        data['joint_control'] = f['joint_control'][:]
        data['joint_states'] = f['joint_states'][:]
        data['timestamp'] = f['timestamp'][:]
        
        # 读取object_states组
        data['object_states/active'] = f['object_states/active'][:]
        data['object_states/passive'] = f['object_states/passive'][:]
        

        '''
        ## test
        print('data: ')
        for key in data.keys():
            print(f'{key}: ', data[key])
        '''

        return data
    
def point_cloud_sampling(point_cloud:np.ndarray, num_points:int, method:str='fps'):
    """
    support different point cloud sampling methods
    point_cloud: (N, 6), xyz+rgb or (N, 3), xyz
    """
    if num_points == 'all': # use all points
        return point_cloud
    
    if point_cloud.shape[0] <= num_points:
        # cprint(f"warning: point cloud has {point_cloud.shape[0]} points, but we want to sample {num_points} points", 'yellow')
        # pad with zeros
        point_cloud_dim = point_cloud.shape[-1]
        point_cloud = np.concatenate([point_cloud, np.zeros((num_points - point_cloud.shape[0], point_cloud_dim))], axis=0)
        return point_cloud

    if method == 'uniform':
        # uniform sampling
        sampled_indices = np.random.choice(point_cloud.shape[0], num_points, replace=False)
        point_cloud = point_cloud[sampled_indices]
    elif method == 'fps':
        # fast point cloud sampling using torch3d
        point_cloud = torch.from_numpy(point_cloud).unsqueeze(0).cuda()
        num_points = torch.tensor([num_points]).cuda()
        # remember to only use coord to sample
        _, sampled_indices = torch3d_ops.sample_farthest_points(points=point_cloud[...,:3], K=num_points)
        point_cloud = point_cloud.squeeze(0).cpu().numpy()
        point_cloud = point_cloud[sampled_indices.squeeze(0).cpu().numpy()]
    else:
        raise NotImplementedError(f"point cloud sampling method {method} not implemented")

    return point_cloud


# ===============================================================

def get_rgb(path):
    img_bgr = cv2.imread(path)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # 将分辨率变成 84 * 84
    img_rgb = cv2.resize(img_rgb, (WIDTH, HEIGHT), interpolation=cv2.INTER_LINEAR)
    
    return img_rgb

def get_depth(path):
    depth_16bit = cv2.imread(path, cv2.IMREAD_ANYDEPTH)  # 保持原始位深

    # print(f"形状: {depth_16bit.shape}")  # (高度, 宽度)
    # print(f"维度: {depth_16bit.ndim}")   # 2（灰度图）
    # print(f"数据类型: {depth_16bit.dtype}")  # uint16 或 float32

    depth_16bit = cv2.resize(depth_16bit, (WIDTH, HEIGHT), interpolation=cv2.INTER_NEAREST)

    return depth_16bit


def get_point_cloud(config, depth_map, color_map=None):
    # Parse config
    width = config['camera']['width']
    height = config['camera']['height']
    max_depth = config['camera']['max_depth']
    scale = config['camera']['scale']
    
    intrinsics = np.array(config['camera']['intrinsics'])
    # Apply scaling
    sx = width / 1280.0
    sy = height / 720.0
    intrinsics[0, 0] *= sx
    intrinsics[1, 1] *= sy
    intrinsics[0, 2] *= sx
    intrinsics[1, 2] *= sy
    
    fx, fy, cx, cy = intrinsics[0, 0], intrinsics[1, 1], intrinsics[0, 2], intrinsics[1, 2]
    
    extrinsic_matrix = np.array(config['camera']['extrinsics'])
    crop_bounds = config['processing']['crop_bounds']

    # 深度裁剪
    depth = depth_map.copy() / scale
    depth[depth > max_depth] = 0.0
    depth = depth.astype(np.float32)

    intrinsic = o3d.camera.PinholeCameraIntrinsic(
        width=width,
        height=height,
        fx=fx,
        fy=fy,
        cx=cx,
        cy=cy
    )

    depth_image = o3d.geometry.Image(depth)

    # If color_map is provided, use RGBD creation to preserve colors
    if color_map is not None:
        # color_map is expected to be HxWx3 uint8 or float
        color_np = color_map
        if color_np.dtype != np.uint8:
            # assume float in [0,1]
            color_np = (np.clip(color_np, 0.0, 1.0) * 255.0).astype(np.uint8)
        color_o3d = o3d.geometry.Image(color_np)
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color_o3d, depth_image, convert_rgb_to_intensity=False, depth_scale=1.0
        )
        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intrinsic)
    else:
        pcd = o3d.geometry.PointCloud.create_from_depth_image(
            depth_image,
            intrinsic
        )
    pcd_depth = o3d.geometry.PointCloud.create_from_depth_image(
        depth_image,
        intrinsic
    )
    T_cw = np.linalg.inv(extrinsic_matrix)
    pcd.transform(T_cw)
    pcd_depth.transform(T_cw)
    
    points = np.asarray(pcd.points)
    colors = None
    if hasattr(pcd, 'colors') and np.asarray(pcd.colors).size > 0:
        # open3d stores colors in 0-1 float
        colors = np.asarray(pcd.colors).astype(np.float32)
    
    x_min, x_max, y_min, y_max, z_min, z_max = crop_bounds
    mask = (points[:, 0] > x_min) & (points[:, 0] < x_max) & \
           (points[:, 1] > y_min) & (points[:, 1] < y_max) & \
           (points[:, 2] > z_min) & (points[:, 2] < z_max)
    points = points[mask]
    if colors is not None:
        colors = colors[mask]

    pc = points
    if colors is not None:
        # ensure colors shape matches and are float32 in [0,1]
        colors = colors.astype(np.float32)
        if colors.max() > 1.1:
            colors = colors / 255.0
        pc = np.concatenate([points, colors], axis=1)  # (N,6)

    # sample
    pc = point_cloud_sampling(pc, 1024, 'fps')

    return pc

def get_state(data):
    pose = np.array(data['ee_states'])
    xyz = pose[:3]
    quat = pose[3:7]
    rot6d = quat_to_rot6d(np.array(quat))
    base_state = np.concatenate((xyz, rot6d))
    if 'gripper_control' in data and data['gripper_control'] is not None:
        gripper = np.array(data['gripper_control']).reshape(-1)
        base_state = np.concatenate((base_state, gripper))
    return base_state

def get_action(data):
    # use ee_control for action and keep gripper control consistent
    pose = np.array(data['ee_control']) if 'ee_control' in data else np.array(data['ee_states'])
    xyz = pose[:3]
    quat = pose[3:7]
    rot6d = quat_to_rot6d(np.array(quat))
    base_action = np.concatenate((xyz, rot6d))
    if 'gripper_control' in data and data['gripper_control'] is not None:
        gripper = np.array(data['gripper_control']).reshape(-1)
        base_action = np.concatenate((base_action, gripper))
    return base_action

def choose(ls, step=STEP):
    return [x for i, x in enumerate(ls) if i % step == 0]

# ===============================================================


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', type=str, default='pick', help='environment to run')
    # parser.add_argument('--num_episodes', type=int, default=100, help='number of episodes to run')
    parser.add_argument('--save_dir', type=str, default='data', help='directory to save data')
    parser.add_argument('--data_dir', type=str, default='data', help='directory to extract data')
    # parser.add_argument('--img_size', type=int, default=84, help='image size')
    # parser.add_argument('--use_point_crop', action='store_true', help='use point crop')
    parser.add_argument('--sample_step', type=int, default=10, help='sampling step for data')
    parser.add_argument('--max_episodes', type=int, default=None, help='maximum number of episodes to process')
    parser.add_argument('--config', type=str, required=True, help='path to config yaml file')
    args = parser.parse_args()
    return args


def render_camera(sim, camera_name="top"):
    img = sim.render(84, 84, camera_name=camera_name)
    return img

def render_high_res(sim, camera_name="top"):
    img = sim.render(1024, 1024, camera_name=camera_name)
    return img


def main():
    args = parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    
    save_dir = os.path.join(args.save_dir, args.env_name+'_zarr_dp3_sim')
    if os.path.exists(save_dir):
        cprint('Data already exists at {}'.format(save_dir), 'red')
        cprint("If you want to overwrite, delete the existing directory first.", "red")
        cprint("Do you want to overwrite? (y/n)", "red")
        # user_input = input()
        user_input = 'y'
        if user_input == 'y':
            cprint('Overwriting {}'.format(save_dir), 'red')
            os.system('rm -rf {}'.format(save_dir))
        else:
            cprint('Exiting', 'red')
            return
    os.makedirs(save_dir, exist_ok=True)
    
    data_dir = args.data_dir

    cprint('Loaded remote controll data from {}'.format(data_dir), 'green')

    total_count = 0
    # img_arrays = []
    point_cloud_arrays = []
    # depth_arrays = []
    state_arrays = []
    action_arrays = []
    episode_ends_arrays = []

    episode_idx = 0
    
    for timestamp in os.listdir(data_dir):
        if args.max_episodes is not None and episode_idx >= args.max_episodes:
            break
        timestamp_dir = os.path.join(data_dir, timestamp)
        print(timestamp_dir)

        for entry in os.listdir(timestamp_dir):
            if args.max_episodes is not None and episode_idx >= args.max_episodes:
                break
            entry_path = Path(timestamp_dir) / entry

            # print(entry_path)
            
            # get sub data
            # Reference convert_zarr.py logic: drive by h5 files
            
            # 1. Find all h5 files
            h5_files = [f for f in os.listdir(entry_path) if f.endswith('.h5')]
            
            # 2. Sort and filter by sample_step
            # Assuming filenames are numbers like "0.h5", "1.h5"
            try:
                h5_files.sort(key=lambda x: int(os.path.splitext(x)[0]))
            except ValueError:
                h5_files.sort() # Fallback
            
            # Filter by sample_step
            h5_files = [f for f in h5_files if int(os.path.splitext(f)[0]) % args.sample_step == 0]
            
            rgb_imgs = []
            depth_imgs = []
            h5_datas = []
            
            for h5_file in h5_files:
                h5_path = entry_path / h5_file
                
                # Read H5
                h5_datas.append(read_hdf5_example(str(h5_path)))
                
                # Construct image paths
                file_stem = os.path.splitext(h5_file)[0]
                
                
                # Depth: {number}_depth.png
                depth_name = f"{file_stem}_depth.png"
                depth_path = entry_path / depth_name
                
                if not depth_path.exists():
                     cprint(f"Warning: Depth not found for {h5_file}", 'yellow')
                
                depth_imgs.append(get_depth(str(depth_path)))

                rgb_name = f"{file_stem}.jpg"
                rgb_path = entry_path / rgb_name
                rgb_imgs.append(get_rgb(str(rgb_path)))

            # Ensure we have enough data
            if len(h5_datas) < 2:
                cprint(f"Skipping {entry_path}: Not enough data ({len(h5_datas)} steps)", 'yellow')
                continue

            # img_arrays_sub = rgb_imgs
            # depth_arrays_sub = depth_imgs
            state_arrays_sub = [get_state(data) for data in h5_datas]
            action_arrays_sub = [get_action(data) for data in h5_datas]
            total_count_sub = len(depth_imgs)
            # Build point clouds with colors when available
            point_cloud_arrays_sub = [get_point_cloud(config, depth, rgb) for depth, rgb in zip(depth_imgs, rgb_imgs)]

            assert len(depth_imgs) == len(state_arrays_sub) == len(action_arrays_sub) == len(point_cloud_arrays_sub), \
                f"Data length mismatch at {entry_path}: depth={len(depth_imgs)}, state={len(state_arrays_sub)}, action={len(action_arrays_sub)}, point_cloud={len(point_cloud_arrays_sub)}"

            # print('depth:', depth_imgs[0].shape)
            # print('rgb:', rgb_imgs[0].shape)
            # print('point cloud:', point_cloud_arrays_sub[0].shape)

            ## check len
            # print('rgb len:', len(rgb_imgs))
            # print('dep len:', len(depth_imgs))
            # print('h5 len:', len(h5_datas))

            # print(len(img_arrays_sub), len(depth_arrays_sub), len(state_arrays_sub), len(action_arrays_sub), total_count_sub, len(point_cloud_arrays_sub))

            total_count += total_count_sub
            episode_ends_arrays.append(deepcopy(total_count)) # the index of the last step of the episode    
            # img_arrays.extend(deepcopy(img_arrays_sub))
            point_cloud_arrays.extend(deepcopy(point_cloud_arrays_sub))
            # depth_arrays.extend(deepcopy(depth_arrays_sub))
            state_arrays.extend(deepcopy(state_arrays_sub))
            action_arrays.extend(deepcopy(action_arrays_sub))

            print('Episode {} at {} over'.format(episode_idx, entry_path)) 

            episode_idx += 1

    # print(episode_ends_arrays)
    print(state_arrays[0])



    ###############################
    # save data
    ###############################
    # create zarr file
    zarr_root = zarr.group(save_dir)
    zarr_data = zarr_root.create_group('data')
    zarr_meta = zarr_root.create_group('meta')
    # save img, state, action arrays into data, and episode ends arrays into meta
    # img_arrays = np.stack(img_arrays, axis=0)
    # if img_arrays.shape[1] == 3: # make channel last
    #     img_arrays = np.transpose(img_arrays, (0,2,3,1))
    state_arrays = np.stack(state_arrays, axis=0)
    point_cloud_arrays = np.stack(point_cloud_arrays, axis=0)
    # depth_arrays = np.stack(depth_arrays, axis=0)
    action_arrays = np.stack(action_arrays, axis=0)
    episode_ends_arrays = np.array(episode_ends_arrays)

    compressor = zarr.Blosc(cname='zstd', clevel=3, shuffle=1)
    # img_chunk_size = (100, img_arrays.shape[1], img_arrays.shape[2], img_arrays.shape[3])
    state_chunk_size = (100, state_arrays.shape[1])
    point_cloud_chunk_size = (100, point_cloud_arrays.shape[1], point_cloud_arrays.shape[2])
    # depth_chunk_size = (100, depth_arrays.shape[1], depth_arrays.shape[2])
    action_chunk_size = (100, action_arrays.shape[1])
    # zarr_data.create_dataset('img', data=img_arrays, chunks=img_chunk_size, dtype='uint8', overwrite=True, compressor=compressor)
    zarr_data.create_dataset('state', data=state_arrays, chunks=state_chunk_size, dtype='float32', overwrite=True, compressor=compressor)
    zarr_data.create_dataset('point_cloud', data=point_cloud_arrays, chunks=point_cloud_chunk_size, dtype='float32', overwrite=True, compressor=compressor)
    # zarr_data.create_dataset('depth', data=depth_arrays, chunks=depth_chunk_size, dtype='float32', overwrite=True, compressor=compressor)
    zarr_data.create_dataset('action', data=action_arrays, chunks=action_chunk_size, dtype='float32', overwrite=True, compressor=compressor)
    zarr_meta.create_dataset('episode_ends', data=episode_ends_arrays, dtype='int64', overwrite=True, compressor=compressor)
    
    
    # print shape
    # cprint(f'img shape: {img_arrays.shape}, range: [{np.min(img_arrays)}, {np.max(img_arrays)}]', 'green')
    cprint(f'point_cloud shape: {point_cloud_arrays.shape}, range: [{np.min(point_cloud_arrays)}, {np.max(point_cloud_arrays)}]', 'green')
    # cprint(f'depth shape: {depth_arrays.shape}, range: [{np.min(depth_arrays)}, {np.max(depth_arrays)}]', 'green')
    cprint(f'state shape: {state_arrays.shape}, range: [{np.min(state_arrays)}, {np.max(state_arrays)}]', 'green')
    cprint(f'action shape: {action_arrays.shape}, range: [{np.min(action_arrays)}, {np.max(action_arrays)}]', 'green')
    cprint(f'Saved zarr file to {save_dir}', 'green')
    
    cprint(f'Saved zarr file to {save_dir}', 'green')
    
    # clean up
    del state_arrays, point_cloud_arrays, action_arrays, episode_ends_arrays
    del zarr_root, zarr_data, zarr_meta
    # del env, expert_agent
    
    
if __name__ == '__main__':
    main()