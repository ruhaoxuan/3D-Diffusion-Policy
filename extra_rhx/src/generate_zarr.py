# usage:
#       bash scripts/vrl3_gen_demonstration_expert.sh door
import matplotlib.pyplot as plt
import argparse
import os
import torch
from termcolor import cprint
from PIL import Image
import zarr
from copy import deepcopy
import numpy as np

from pathlib import Path

import re
from typing import List, Union
from pathlib import Path

import h5py
import cv2
import open3d as o3d
import pytorch3d.ops as torch3d_ops

# constant
FX, FY, CX, CY = 72.70, 72.72, 42.0, 42.0
STEP = 1
WIDTH = 84
HEIGHT = 84

def extract_with_pattern(dir_path: Union[str, Path],
                         pattern: str = r'.*\.jpg$',
                         recursive: bool = False,
                         sort_key: str = 'numeric') -> List[Path]:
    """
    从目录中提取符合正则 pattern 的文件列表并排序。
    参数:
      dir_path: 目录路径（str 或 Path）
      pattern: 正则表达式，默认匹配所有 .jpg 文件（不区分大小写可在 pattern 中添加 (?i)）
      recursive: 是否递归子目录（默认 False）
      sort_key: 'name'（按文件名字符串排序）或 'numeric'（按文件名前数字排序，如果无法解析回退到 name）
    返回:
      符合条件的 Path 列表（已排序）
    示例:
      jpgs = extract_with_pattern(path, pattern=r'^\d+_frame_\d+\.jpg$')
    """
    p = Path(dir_path)
    if not p.exists():
        return []

    regex = re.compile(pattern)
    if recursive:
        it = p.rglob('*')
    else: 
        it = p.iterdir()

    files = [f.name for f in it if f.is_file() and regex.match(f.name)]
    if sort_key == 'numeric':
        def _num_key(fp: str):
            name = fp
            m = re.match(r'(\d+)', name)
            if m:
                return int(m.group(1))
            return name
        try:
            files.sort(key=_num_key)
        except Exception:
            files.sort()
    else:
        files.sort()
    return files

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

def depth_info(path):
    depth = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    print(f"数据类型: {depth.dtype}")
    print(f"形状: {depth.shape}")
    
    # 根据数据类型判断位深
    dtype_to_bit = {
        'uint8': 8,
        'uint16': 16,
        'int16': 16,
        'float32': 32,
        'float64': 64
    }
    
    bit_depth = dtype_to_bit.get(str(depth.dtype), "未知")
    print(f"位深: {bit_depth}位")
    
    # 显示数值范围
    print(f"最小值: {depth.min()}, 最大值: {depth.max()}")

def get_depth(path):
    depth_16bit = cv2.imread(path, cv2.IMREAD_ANYDEPTH)  # 保持原始位深

    # print(f"形状: {depth_16bit.shape}")  # (高度, 宽度)
    # print(f"维度: {depth_16bit.ndim}")   # 2（灰度图）
    # print(f"数据类型: {depth_16bit.dtype}")  # uint16 或 float32

    depth_16bit = cv2.resize(depth_16bit, (WIDTH, HEIGHT), interpolation=cv2.INTER_NEAREST)

    return depth_16bit


def get_point_cloud(depth_map):
    fx, fy, cx, cy = FX, FY, CX, CY

    intrinsic = o3d.camera.PinholeCameraIntrinsic(
        width=WIDTH,
        height=HEIGHT,
        fx=fx,
        fy=fy,
        cx=cx,
        cy=cy
    )

    depth_image = o3d.geometry.Image(depth_map)

    pcd = o3d.geometry.PointCloud.create_from_depth_image(
        depth_image,
        intrinsic
    )

    points = np.asarray(pcd.points)

    # print(points.shape)
    # return
    
    points = point_cloud_sampling(points, 512, 'fps')

    return points

    ## 还没考虑降采样

def get_state(data):
    state = np.array(np.concatenate((data['ee_states'], data['gripper_control'])))

    return state

def get_action(data):
    return get_state(data)

def choose(ls, step=STEP):
    return [x for i, x in enumerate(ls) if i % step == 0]

# ===============================================================


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', type=str, default='pick', help='environment to run')
    parser.add_argument('--num_episodes', type=int, default=100, help='number of episodes to run')
    parser.add_argument('--save_dir', type=str, default='data', help='directory to save data')
    parser.add_argument('--data_dir', type=str, default='data', help='directory to extract data')
    parser.add_argument('--img_size', type=int, default=84, help='image size')
    parser.add_argument('--use_point_crop', action='store_true', help='use point crop')
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
    
    num_episodes = args.num_episodes
    save_dir = os.path.join(args.save_dir, 'final_'+args.env_name+'_sim.zarr')
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
    img_arrays = []
    point_cloud_arrays = []
    depth_arrays = []
    state_arrays = []
    action_arrays = []
    episode_ends_arrays = []

    episode_idx = 0
    
    for timestamp in os.listdir(data_dir):
        timestamp_dir = os.path.join(data_dir, timestamp)
        print(timestamp_dir)

        for entry in os.listdir(timestamp_dir):
            entry_path = Path(timestamp_dir) / entry

            # print(entry_path)
            
            # get sub data
            rgb_files = extract_with_pattern(entry_path, r'^\d+\.jpg$')
            rgb_files = choose(rgb_files)

            rgb_imgs = [get_rgb(entry_path / image) for image in rgb_files]
            
            # print(entry_path / rgb_files[0]) ##

            depth_files = extract_with_pattern(entry_path, r'^\d+_depth.png$')
            depth_files = choose(depth_files)

            # info = depth_info(entry_path / depth_files[0])
            # test_depth = get_depth(entry_path / depth_files[0])

            depth_imgs = [get_depth(entry_path / image) for image in depth_files]

            h5_files = extract_with_pattern(entry_path, r'^.+\.h5$')
            h5_files = choose(h5_files, STEP * 5)

            # print(rgb_files[:10])
            # print(depth_files[:10])
            # print(h5_files[:10])
            # return

            h5_datas = [read_hdf5_example(entry_path / h) for h in h5_files]

            img_arrays_sub = rgb_imgs[:-1]
            depth_arrays_sub = depth_imgs[:-1]
            state_arrays_sub = [get_state(data) for data in h5_datas[:-1]]
            action_arrays_sub = [get_action(data) for data in h5_datas[1:]]
            total_count_sub = len(rgb_imgs[:-1])
            point_cloud_arrays_sub = [get_point_cloud(depth) for depth in depth_imgs[:-1]]

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
            img_arrays.extend(deepcopy(img_arrays_sub))
            point_cloud_arrays.extend(deepcopy(point_cloud_arrays_sub))
            depth_arrays.extend(deepcopy(depth_arrays_sub))
            state_arrays.extend(deepcopy(state_arrays_sub))
            action_arrays.extend(deepcopy(action_arrays_sub))

            print('Episode {} at {} over'.format(episode_idx, entry_path)) 

            episode_idx += 1

    # print(episode_ends_arrays)
    print(state_arrays[0])

    # loop over episodes
    '''
    minimal_episode_length = 100
    episode_idx = 0
    while episode_idx < num_episodes:
        env = create_env()
        time_step = env.reset()
        input_obs_visual = time_step.observation # (3n,84,84), unit8
        input_obs_sensor = time_step.observation_sensor # float32, door(24,)q        

        total_reward = 0.
        n_goal_achieved_total = 0.
        step_count = 0
        
        img_arrays_sub = []
        point_cloud_arrays_sub = []
        depth_arrays_sub = []
        state_arrays_sub = []
        action_arrays_sub = []act
        total_count_sub = 0
        
        while (not time_step.last()) or step_count < minimal_episode_length:
            with torch.no_grad(), utils.eval_mode(expert_agent):
                input_obs_visual = time_step.observation
                input_obs_sensor = time_step.observation_sensor
                # cam: top, vil_camera, fixed
                # vrl3_input = render_camera(env.env._env.sim, camera_name="top").transpose(2,0,1).copy() # (3,84,84)
                    
                action = expert_agent.act(obs=input_obs_visual, step=0,
                                        eval_mode=True, 
                                        obs_sensor=input_obs_sensor) # (28,) float32
                
                if args.not_use_multi_view:
                    input_obs_visual = input_obs_visual[:3] # (3,84,84)
                

                        
                # save data
                total_count_sub += 1
                img_arrays_sub.append(input_obs_visual)
                state_arrays_sub.append(input_obs_sensor)
                action_arrays_sub.append(action)
                point_cloud_arrays_sub.append(time_step.observation_pointcloud)
                depth_arrays_sub.append(time_step.observation_depth)
                
            time_step = env.step(action)
            obs = time_step.observation # np array, (3,84,84)
            obs = obs[:3] if obs.shape[0] > 3 else obs # (3,84,84)
            n_goal_achieved_total += time_step.n_goal_achieved
            total_reward += time_step.reward
            step_count += 1
            
        if n_goal_achieved_total < 10.:
            cprint(f"Episode {episode_idx} has {n_goal_achieved_total} goals achieved and {total_reward} reward. Discarding.", 'red')
        else:
            total_count += total_count_sub
            episode_ends_arrays.append(deepcopy(total_count)) # the index of the last step of the episode    
            img_arrays.extend(deepcopy(img_arrays_sub))
            point_cloud_arrays.extend(deepcopy(point_cloud_arrays_sub))
            depth_arrays.extend(deepcopy(depth_arrays_sub))
            state_arrays.extend(deepcopy(state_arrays_sub))
            action_arrays.extend(deepcopy(action_arrays_sub))
            print('Episode: {}, Reward: {}, Goal Achieved: {}'.format(episode_idx, total_reward, n_goal_achieved_total)) 
            episode_idx += 1

    # tracemalloc.stop()

    '''

    # return

    ###############################
    # save data
    ###############################
    # create zarr file
    zarr_root = zarr.group(save_dir)
    zarr_data = zarr_root.create_group('data')
    zarr_meta = zarr_root.create_group('meta')
    # save img, state, action arrays into data, and episode ends arrays into meta
    img_arrays = np.stack(img_arrays, axis=0)
    if img_arrays.shape[1] == 3: # make channel last
        img_arrays = np.transpose(img_arrays, (0,2,3,1))
    state_arrays = np.stack(state_arrays, axis=0)
    point_cloud_arrays = np.stack(point_cloud_arrays, axis=0)
    depth_arrays = np.stack(depth_arrays, axis=0)
    action_arrays = np.stack(action_arrays, axis=0)
    episode_ends_arrays = np.array(episode_ends_arrays)

    compressor = zarr.Blosc(cname='zstd', clevel=3, shuffle=1)
    img_chunk_size = (100, img_arrays.shape[1], img_arrays.shape[2], img_arrays.shape[3])
    state_chunk_size = (100, state_arrays.shape[1])
    point_cloud_chunk_size = (100, point_cloud_arrays.shape[1], point_cloud_arrays.shape[2])
    depth_chunk_size = (100, depth_arrays.shape[1], depth_arrays.shape[2])
    action_chunk_size = (100, action_arrays.shape[1])
    zarr_data.create_dataset('img', data=img_arrays, chunks=img_chunk_size, dtype='uint8', overwrite=True, compressor=compressor)
    zarr_data.create_dataset('state', data=state_arrays, chunks=state_chunk_size, dtype='float32', overwrite=True, compressor=compressor)
    zarr_data.create_dataset('point_cloud', data=point_cloud_arrays, chunks=point_cloud_chunk_size, dtype='float32', overwrite=True, compressor=compressor)
    zarr_data.create_dataset('depth', data=depth_arrays, chunks=depth_chunk_size, dtype='float32', overwrite=True, compressor=compressor)
    zarr_data.create_dataset('action', data=action_arrays, chunks=action_chunk_size, dtype='float32', overwrite=True, compressor=compressor)
    zarr_meta.create_dataset('episode_ends', data=episode_ends_arrays, dtype='int64', overwrite=True, compressor=compressor)
    
    
    # print shape
    cprint(f'img shape: {img_arrays.shape}, range: [{np.min(img_arrays)}, {np.max(img_arrays)}]', 'green')
    cprint(f'point_cloud shape: {point_cloud_arrays.shape}, range: [{np.min(point_cloud_arrays)}, {np.max(point_cloud_arrays)}]', 'green')
    cprint(f'depth shape: {depth_arrays.shape}, range: [{np.min(depth_arrays)}, {np.max(depth_arrays)}]', 'green')
    cprint(f'state shape: {state_arrays.shape}, range: [{np.min(state_arrays)}, {np.max(state_arrays)}]', 'green')
    cprint(f'action shape: {action_arrays.shape}, range: [{np.min(action_arrays)}, {np.max(action_arrays)}]', 'green')
    cprint(f'Saved zarr file to {save_dir}', 'green')
    
    cprint(f'Saved zarr file to {save_dir}', 'green')
    
    # clean up
    del img_arrays, state_arrays, point_cloud_arrays, action_arrays, episode_ends_arrays
    del zarr_root, zarr_data, zarr_meta
    # del env, expert_agent
    
    
if __name__ == '__main__':
    main()