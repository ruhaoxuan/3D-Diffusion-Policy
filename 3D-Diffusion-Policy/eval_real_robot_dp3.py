"""
Evaluation script for DP3 on Real Robot (or Genesis Sim).
Adapted from eval_real_robot_continue.py
"""

import sys
import os
import pathlib
import time
import click
import cv2
import numpy as np
import torch
import dill
import hydra
from omegaconf import OmegaConf
from termcolor import cprint
from tqdm import tqdm
from pynput import keyboard
import rospy
from sensor_msgs.msg import Image
from collections import deque
import scipy.spatial.transform as st
import termios

import open3d as o3d
import yaml
sys.path.append('..')
# Do NOT import hardcoded constants from scripts.constant here.
# The script will read required camera/sensor parameters from a YAML config passed at runtime.
import visualizer
import json

# Add paths
# TAMP/policy/3D-Diffusion-Policy/3D-Diffusion-Policy/eval_real_robot_dp3.py
# CURRENT_DIR = pathlib.Path(__file__).parent
# ROOT_DIR = str(CURRENT_DIR.parent.parent) # TAMP/policy
# sys.path.append(os.path.join(ROOT_DIR, 'diffusion_policy_tamp'))
# sys.path.append(str(CURRENT_DIR))
sys.path.append("../../policy/diffusion_policy_tamp")
from train import TrainDP3Workspace
from diffusion_policy_3d.common.pytorch_util import dict_apply
try:
    from diffusion_policy.model.common.rotation_transformer import RotationTransformer
    from diffusion_policy.device.franka_sim import FrankaGenesisEnvWrapper
    from diffusion_policy.real_world.ensemble import EnsembleBuffer
    from diffusion_policy.dataset.utils.process_obs import convert_tcp_data_to_camera
except ImportError as e:
    cprint(f"Error importing from diffusion_policy_tamp: {e}", "red")
    cprint("Make sure you are in the correct environment and paths are set.", "red")
    raise e

OmegaConf.register_new_resolver("eval", eval, replace=True)

# -----------------------------------------------------------------------------
# Helper Classes and Functions
# -----------------------------------------------------------------------------

def clear_input_buffer():
    termios.tcflush(sys.stdin, termios.TCIFLUSH)

class SimCameraRGBD:
    def __init__(self, color_topic='/genesis/color_image', depth_topic='/genesis/depth_image', mask_topic='/genesis/mask_image'):
        self.color_img = None
        self.depth_img = None
        self.mask_img = None
        
        # Initialize node if not already initialized
        # try:
        #     rospy.init_node('dp3_eval_node', anonymous=True)
        # except rospy.exceptions.ROSException:
        #     pass

        rospy.Subscriber(color_topic, Image, self.color_callback)
        rospy.Subscriber(depth_topic, Image, self.depth_callback)
        rospy.Subscriber(mask_topic, Image, self.mask_callback)
        
        # Wait for images
        cprint("Waiting for camera images...", "yellow")
        while self.color_img is None: # or self.depth_img is None: # Allow depth to be missing for now if topic is wrong
            time.sleep(0.1)
        cprint("Camera connected.", "green")

    def color_callback(self, msg):
        height = msg.height
        width = msg.width
        encoding = msg.encoding
        if encoding in ['rgb8', 'bgr8']:
            channels = 3
        elif encoding == 'mono8':
            channels = 1
        else:
            return
        image_data = np.frombuffer(msg.data, dtype=np.uint8)
        img = image_data.reshape((height, width, channels))
        if encoding == 'bgr8':
            img = img[:, :, ::-1]
        self.color_img = img

    def depth_callback(self, msg):
        height = msg.height
        width = msg.width
        # Assuming 16-bit depth (mm) or 32-bit float (m)
        if msg.encoding == '16UC1':
            dtype = np.uint16
            scale = 0.001 # mm to m
        elif msg.encoding == '32FC1':
            dtype = np.float32
            scale = 1.0
        else:
            return
        
        image_data = np.frombuffer(msg.data, dtype=dtype)
        img = image_data.reshape((height, width))
        self.depth_img = img.astype(np.float32) * scale
    
    def mask_callback(self, msg):
        height = msg.height
        width = msg.width
        if msg.encoding == '32FC1':
            dtype = np.uint8
            scale = 1
        else:
            return
        
        image_data = np.frombuffer(msg.data, dtype=dtype)
        img = image_data.reshape((height, width))
        self.mask_img = img.astype(np.uint8) * scale

    def get_rgbd_image(self):
        return self.color_img, self.depth_img, self.mask_img

quat2rot6d_transformer = RotationTransformer(from_rep='quaternion', to_rep="rotation_6d")
rot6d_quat_transformer = RotationTransformer(from_rep='rotation_6d', to_rep="quaternion")

def transform_ee_pose_frame(ee_pose: np.ndarray, frame: str) -> np.ndarray:
    if frame == "camera":
        return convert_tcp_data_to_camera(ee_pose)
    elif frame == "base":
        return ee_pose
    else:
        raise ValueError(f"Unsupported frame type: {frame}")

class DP3Agent:
    def __init__(self, obs_num, gripper, frame="base", **kwargs):
        self.obs_num = obs_num
        self.frame = frame
        self.gripper = gripper
        
        self.rgb_buffer = deque(maxlen=obs_num)
        self.depth_buffer = deque(maxlen=obs_num)
        self.mask_buffer = deque(maxlen=obs_num)
        self.pose_buffer = deque(maxlen=obs_num)
        
        
        self.arm = FrankaGenesisEnvWrapper(control_mode="joint", gripper="panda" if gripper else None, gripper_init_state="open")
        self.camera = SimCameraRGBD()
        self._warmup_camera()
        self._fill_initial_buffer()

    def _warmup_camera(self):
        for _ in range(10):
            self.camera.get_rgbd_image()
            time.sleep(0.1)

    def _fill_initial_buffer(self):
        for _ in range(self.obs_num):
            self._add_single_observation()
    
    def reset_buffer(self):
        self.rgb_buffer.clear()
        self.depth_buffer.clear()
        self.pose_buffer.clear()
        self._fill_initial_buffer()

    def _add_single_observation(self):
        color, depth, mask = self.camera.get_rgbd_image()
        
        # Resize if needed (DP3 usually expects specific size, e.g. 84x84 or 128x128)
        # For now, keep original or resize to 84x84 as common in DP3
        # You might need to adjust this based on your model config
        target_size = (1280, 720) 
        if color is not None:
            color = cv2.resize(color, target_size, interpolation=cv2.INTER_LINEAR)
        if depth is not None:
            depth = cv2.resize(depth, target_size, interpolation=cv2.INTER_NEAREST)
        else:
            # Mock depth if missing
            depth = np.zeros(target_size, dtype=np.float32)
        if mask is not None:
            mask = cv2.resize(mask, target_size, interpolation=cv2.INTER_NEAREST)
        else:
            # Mock mask if missing
            mask = np.zeros(target_size, dtype=np.uint8)

        # arm_ee_pose = self.arm.get_tcp_position()
        # transformed_ee_pose = transform_ee_pose_frame(arm_ee_pose, self.frame)
        # xyz, quat = transformed_ee_pose[:3], transformed_ee_pose[3:7]
        # rot6d = quat2rot6d_transformer.forward(np.array(quat))
        # pose = np.concatenate([xyz, rot6d])
        arm_joint_pos = self.arm.get_arm_position()
        pose = np.array(arm_joint_pos[:-2])
        # print(pose.shape)
        
        if self.gripper:
            gripper_width = self.arm.get_gripper_position()
            pose = np.hstack([pose, gripper_width])

        self.rgb_buffer.append(color)
        self.depth_buffer.append(depth)
        self.mask_buffer.append(mask)
        self.pose_buffer.append(pose)

    def get_observation(self):
        self._add_single_observation()
        return {
            "rgb": np.array(self.rgb_buffer),
            "depth": np.array(self.depth_buffer),
            "mask": np.array(self.mask_buffer),
            "pose": np.array(self.pose_buffer),
            "raw_img": self.rgb_buffer[-1] # For visualization
        }
    
    def set_tcp_pose(self, pose):
        # pose: 9-dim (xyz + rot6d)
        # xyz = pose[:3]
        # rot6d = pose[3:9]
        # quat = rot6d_quat_transformer.forward(rot6d)
        
        # Transform back if needed? Assuming input is in 'frame'
        # If frame is camera, we need to transform back to base for robot control
        # But FrankaGenesisEnvWrapper usually expects base frame.
        # If model outputs in camera frame, we need inverse transform.
        # For now assume model outputs in base frame or we handle it.
        
        # If the model was trained with camera frame observations, it likely outputs camera frame actions.
        # But here we just pass it to arm.
        
        # target_pose = np.concatenate([xyz, quat])
        # self.arm.set_tcp_pose(target_pose)
        # self.arm.move_ee(target_pose)
        target_pose = pose
        self.arm.move_joint(target_pose)

    def set_tcp_gripper(
        self, gripper, blocking=False
    ):
        self.arm.move_gripper(gripper > 0.5)

# -----------------------------------------------------------------------------
# Point Cloud Generation
# -----------------------------------------------------------------------------

import pytorch3d.ops as torch3d_ops

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



def get_point_cloud(config, depth_map, color_map=None, mask_map=None):

    # print(np.unique(mask_map))

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
    # depth[depth > max_depth] = 0.0
    # check if the mask_map is all zero
    # print("Mask map unique values:", np.unique(mask_map) if mask_map is not None else "No mask map provided")
    if mask_map is not None:
        depth[mask_map == 0] = 0.0
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
    
    # x_min, x_max, y_min, y_max, z_min, z_max = crop_bounds
    # mask = (points[:, 0] > x_min) & (points[:, 0] < x_max) & \
    #        (points[:, 1] > y_min) & (points[:, 1] < y_max) & \
    #        (points[:, 2] > z_min) & (points[:, 2] < z_max)
    # points = points[mask]
    # if colors is not None:
    #     colors = colors[mask]

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

def get_real_obs_dict(agent_obs, shape_meta, sensor_config: dict):
    # agent_obs: {'rgb': (T, H, W, 3), 'depth': (T, H, W), 'pose': (T, D)}
    # Output: {'point_cloud': (T, N, 3 or 6), 'agent_pos': (T, D)}
    
    rgb = agent_obs['rgb']
    depth = agent_obs['depth']
    mask = agent_obs['mask']
    pose = agent_obs['pose']
    
    T, H, W, _ = rgb.shape
    
    point_clouds = []
    for i in range(T):
        # pass color frame so get_point_cloud can produce colored point clouds
        pc = get_point_cloud(sensor_config, depth[i], rgb[i], mask[i])
        point_clouds.append(pc)
        
    point_clouds = np.array(point_clouds) # (T, N, 3)
    
    return {
        'point_cloud': point_clouds.astype(np.float32),
        'agent_pos': pose.astype(np.float32)
    }


def visualize_pointcloud_and_target(obs_or_points, sensor_config, action, sphere_radius=0.02, frame_size=0.05, window_title="PointCloud Preview"):
    """
    Visualize an observed point cloud (or an observation dict) and a predicted target action position.

    Args:
        obs_or_points: Either an agent observation dict (as returned by Agent.get_observation())
                       or a numpy array of points with shape (N,3) or (T,N,3).
        sensor_config: YAML-loaded camera/sensor config (used if obs_or_points is an obs dict).
        action: array-like action vector where first 3 elements are xyz.
        sphere_radius: radius of the marker sphere (meters).
        frame_size: size of coordinate frame at the target.
        window_title: Title for the Open3D window.
    """
    # Resolve points
    points = None
    try:
        if isinstance(obs_or_points, dict) and 'depth' in obs_or_points:
            # Convert observation -> point cloud using existing helper
            obs_dict = get_real_obs_dict(obs_or_points, None, sensor_config)
            pc = obs_dict['point_cloud']
            # pc shape could be (T, N, 3)
            if isinstance(pc, np.ndarray) and pc.ndim == 3:
                points = pc[-1]
            # if points have color channels (N,6), drop to XYZ for visualization
            if points is not None and points.ndim == 2 and points.shape[1] >= 6:
                points = points[:, :3]
            else:
                points = pc
        elif isinstance(obs_or_points, np.ndarray):
            arr = obs_or_points
            if arr.ndim == 3:
                points = arr[-1]
            elif arr.ndim == 2:
                points = arr
            else:
                points = arr.reshape(-1, 3)
        else:
            # Unknown type: try to coerce
            points = np.asarray(obs_or_points)
            if points.ndim == 3:
                points = points[-1]
    except Exception:
        points = None

    if points is None or (isinstance(points, np.ndarray) and points.size == 0):
        # use a single placeholder point at origin to avoid empty pcd issues
        points = np.zeros((1, 3), dtype=np.float64)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))
    # If obs_or_points contains color channels, assign them to pcd.colors (expects 0..1 floats)
    try:
        # Case A: obs_or_points is a numpy array with color channels (N,6)
        if isinstance(obs_or_points, np.ndarray) and obs_or_points.ndim == 2 and obs_or_points.shape[1] >= 6:
            cols = obs_or_points[-1][:, 3:6]
            # If cols in 0..255, scale down
            if cols.max() > 1.1:
                cols = (cols / 255.0).astype(np.float64)
            pcd.colors = o3d.utility.Vector3dVector(cols.astype(np.float64))

        # Case B: obs_or_points is an observation dict (has 'depth'/'rgb')
        elif isinstance(obs_or_points, dict) and 'depth' in obs_or_points and 'rgb' in obs_or_points:
            try:
                obs_dict = get_real_obs_dict(obs_or_points, None, sensor_config)
                pc_full = obs_dict.get('point_cloud', None)
                if isinstance(pc_full, np.ndarray) and pc_full.ndim == 3:
                    last_pc = pc_full[-1]
                    if last_pc.ndim == 2 and last_pc.shape[1] >= 6:
                        cols = last_pc[:, 3:6]
                        if cols.max() > 1.1:
                            cols = (cols / 255.0).astype(np.float64)
                        pcd.colors = o3d.utility.Vector3dVector(cols.astype(np.float64))
            except Exception:
                # silently ignore color assignment errors and fall back to no colors
                pass
    except Exception:
        pass

    # action -> xyz
    act = np.asarray(action)
    if act.size >= 3:
        xyz = act[:3].astype(np.float64)
    else:
        xyz = np.zeros(3, dtype=np.float64)

    # Sphere marker and coordinate frame
    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=sphere_radius)
    sphere.compute_vertex_normals()
    sphere.paint_uniform_color([1.0, 0.0, 0.0])
    sphere.translate(xyz)

    frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=frame_size, origin=xyz)

    # Optionally add axes at origin for reference
    origin_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=frame_size*2)

    # Show
    try:
        o3d.visualization.draw_geometries([pcd, sphere, frame, origin_frame], window_name=window_title)
    except Exception as e:
        # In some headless environments Open3D visualization can fail; re-raise to let caller handle
        raise e

# -----------------------------------------------------------------------------
# Main Loop
# -----------------------------------------------------------------------------

interrupted = False
def _keyboard_signal_handler(key):
    global interrupted
    try:
        if key.char == 't':
            interrupted = True
            cprint("\nStopping...", "yellow")
    except AttributeError:
        pass

keyboard_listener = keyboard.Listener(on_press=_keyboard_signal_handler)
keyboard_listener.start()

def run_single_episode(agent:DP3Agent, policy, cfg, device, max_duration, gripper, output, ctrl_interval, sensor_config: dict):
    global interrupted
    global keyboard_listener
    
    try:
        agent.arm.home_robot()
        input("Press Enter to start.")
    except Exception as e:
        cprint(f"Home failed: {e}", "red")
        
    print("Warming up policy inference")
    agent.reset_buffer()
    
    # Warmup
    obs = agent.get_observation()
    with torch.no_grad():
        policy.reset()
        obs_dict_np = get_real_obs_dict(obs, cfg.task.shape_meta, sensor_config)
        obs_dict = dict_apply(obs_dict_np, lambda x: torch.from_numpy(x).unsqueeze(0).to(device))
        result = policy.predict_action(obs_dict)
        del result

    ensemble_buffer = EnsembleBuffer(mode="avg")
    
    with torch.inference_mode():
        print("Start!")
        raw_imgs = []
        
        for t in tqdm(range(max_duration), desc="evaluating"):
            if not keyboard_listener.is_alive():
                keyboard_listener = keyboard.Listener(on_press=_keyboard_signal_handler)
                keyboard_listener.start()
            if interrupted:
                break
            
            if t % 5 == 0: # Inference frequency
                obs = agent.get_observation()
                obs_dict_np = get_real_obs_dict(obs, cfg.task.shape_meta, sensor_config)
                obs_dict = dict_apply(obs_dict_np, lambda x: torch.from_numpy(x).unsqueeze(0).to(device))

                result = policy.predict_action(obs_dict)
                raw_action = result['action'][0].detach().to('cpu').numpy()
                ensemble_buffer.add_action(raw_action, t)
            
            step_action = ensemble_buffer.get_action()
            # if t % 5 == 0:
            #     visualize_pointcloud_and_target(obs, sensor_config, step_action)
            if step_action is None:
                continue
                
            # Execute action: 9-dim (xyz + rot6d) or 10-dim (+gripper)
            print(f"Step {t}: Executing action {step_action[-1]}")
            agent.set_tcp_pose(step_action[:10])
            if gripper and len(step_action) > 10:
                agent.set_tcp_gripper(step_action[10])
            
            # Rate limiting to slow down execution
            if ctrl_interval is not None and ctrl_interval > 0:
                time.sleep(ctrl_interval)
                
            # Record
            if output is not None:
                raw_imgs.append(obs["raw_img"])
                
        if output is not None and raw_imgs:
            for idx, img in enumerate(raw_imgs):
                cv2.imwrite(os.path.join(output, f"rgb_{idx:05d}.png"), img)
            cprint(f"Images saved to {output}", "cyan")
            
    return not interrupted

@click.command()
@click.option('--ckpt', '-i', required=True, help='Path to checkpoint')
@click.option('--output', '-o', default=None, type=str, help='Directory to save recording')
@click.option('--max_duration', '-md', default=1000, help='Max duration in steps.')
@click.option('--gripper', '-g', is_flag=True, default=False, type=bool, help='Enable gripper control')
@click.option('--continuous', '-c', is_flag=True, default=False, type=bool, help='Enable continuous testing mode')
@click.option('--ctrl_hz', default=5.0, type=float, help='Control frequency (Hz). Set 0 to disable rate limiting')
@click.option('--config', required=True, type=str, help='Path to camera/sensor YAML config (required)')
def main(ckpt, output, max_duration, gripper, continuous, ctrl_hz, config):
    global interrupted
    
    # Load Checkpoint
    payload = torch.load(open(ckpt, 'rb'), pickle_module=dill)
    cfg = payload['cfg']
    
    # Load sensor config from YAML (must be provided)
    with open(config, 'r') as f:
        sensor_config = yaml.safe_load(f)
    
    # Setup Workspace
    workspace = TrainDP3Workspace(cfg)
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)
    
    # Get Policy
    policy = workspace.model
    if cfg.training.use_ema:
        policy = workspace.ema_model
    
    device = torch.device('cuda')
    policy.eval().to(device)
    
    # Setup Agent
    n_obs_steps = cfg.n_obs_steps
    
    # Determine observation params from cfg (rot6d only)
    try:
        agent_pos_shape = cfg.task.shape_meta.obs.agent_pos.shape
        state_dim = agent_pos_shape[0]
    except:
        state_dim = 10 # Default fallback

    if state_dim == 10:
        model_has_gripper = False
    elif state_dim == 11:
        model_has_gripper = True
    else:
        raise ValueError(f"Unsupported state_dim {state_dim} for rot6d-only setup. Expected 9 or 10.")
    
    cprint(f"Detected state dim: {state_dim}, use_rot6d: True, model_has_gripper: {model_has_gripper}", "yellow")
    
    agent = DP3Agent(obs_num=n_obs_steps, gripper=model_has_gripper)

    ctrl_interval = 1.0 / ctrl_hz if ctrl_hz > 0 else 0.0
    
    if continuous:
        episode_count = 0
        cprint("Press T to terminate this test.", "cyan")
        while True:
            episode_count += 1
            interrupted = False
            cprint(f"\nTesting No. {episode_count}", "blue")
            
            episode_output = None
            if output is not None:
                episode_output = os.path.join(output, f"episode_{episode_count:03d}")
                os.makedirs(episode_output, exist_ok=True)
                
            success = run_single_episode(agent, policy, cfg, device, max_duration, gripper, episode_output, ctrl_interval, sensor_config)
            
            if success:
                cprint(f"Test No. {episode_count} finished", "green")
            else:
                cprint(f"Test No. {episode_count} stopped", "yellow")
                
            try:
                clear_input_buffer()
                response = input("continue? (y/n)")
                if response.strip().lower() in ['n', 'no', 'exit', 'quit']:
                    break
            except KeyboardInterrupt:
                break
    else:
        if output is not None:
            os.makedirs(output, exist_ok=True)
    run_single_episode(agent, policy, cfg, device, max_duration, gripper, output, ctrl_interval, sensor_config)
        
    agent.arm.home_robot()
    print("Goodbye.")

if __name__ == '__main__':
    main()
