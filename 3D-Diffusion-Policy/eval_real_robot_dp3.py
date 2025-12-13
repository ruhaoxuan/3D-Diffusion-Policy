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
    def __init__(self, color_topic='/genesis/color_image', depth_topic='/genesis/depth_image'):
        self.color_img = None
        self.depth_img = None
        
        # Initialize node if not already initialized
        # try:
        #     rospy.init_node('dp3_eval_node', anonymous=True)
        # except rospy.exceptions.ROSException:
        #     pass

        rospy.Subscriber(color_topic, Image, self.color_callback)
        rospy.Subscriber(depth_topic, Image, self.depth_callback)
        
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

    def get_rgbd_image(self):
        return self.color_img, self.depth_img

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
    def __init__(self, obs_num, gripper, frame="base", use_rot6d=True, **kwargs):
        self.obs_num = obs_num
        self.frame = frame
        self.gripper = gripper
        self.use_rot6d = use_rot6d
        
        self.rgb_buffer = deque(maxlen=obs_num)
        self.depth_buffer = deque(maxlen=obs_num)
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
        color, depth = self.camera.get_rgbd_image()
        
        # Resize if needed (DP3 usually expects specific size, e.g. 84x84 or 128x128)
        # For now, keep original or resize to 84x84 as common in DP3
        # You might need to adjust this based on your model config
        target_size = (84, 84) 
        if color is not None:
            color = cv2.resize(color, target_size, interpolation=cv2.INTER_LINEAR)
        if depth is not None:
            depth = cv2.resize(depth, target_size, interpolation=cv2.INTER_NEAREST)
        else:
            # Mock depth if missing
            depth = np.zeros(target_size, dtype=np.float32)

        arm_ee_pose = self.arm.get_tcp_position()
        transformed_ee_pose = transform_ee_pose_frame(arm_ee_pose, self.frame)
        xyz, quat = transformed_ee_pose[:3], transformed_ee_pose[3:7]
        if self.use_rot6d:
            rot6d = quat2rot6d_transformer.forward(np.array(quat))
            pose = np.concatenate([xyz, rot6d])
        else:
            pose = np.concatenate([xyz, quat])
        
        if self.gripper:
            gripper_width = self.arm.get_gripper_position()
            pose = np.hstack([pose, gripper_width])

        self.rgb_buffer.append(color)
        self.depth_buffer.append(depth)
        self.pose_buffer.append(pose)

    def get_observation(self):
        self._add_single_observation()
        return {
            "rgb": np.array(self.rgb_buffer),
            "depth": np.array(self.depth_buffer),
            "pose": np.array(self.pose_buffer),
            "raw_img": self.rgb_buffer[-1] # For visualization
        }
    
    def set_tcp_pose(self, pose):
        # pose: 9-dim (xyz + rot6d) or 7-dim (xyz + quat)
        xyz = pose[:3]
        if self.use_rot6d:
            rot6d = pose[3:9]
            quat = rot6d_quat_transformer.forward(rot6d)
        else:
            quat = pose[3:7]
        
        # Transform back if needed? Assuming input is in 'frame'
        # If frame is camera, we need to transform back to base for robot control
        # But FrankaGenesisEnvWrapper usually expects base frame.
        # If model outputs in camera frame, we need inverse transform.
        # For now assume model outputs in base frame or we handle it.
        
        # If the model was trained with camera frame observations, it likely outputs camera frame actions.
        # But here we just pass it to arm.
        
        target_pose = np.concatenate([xyz, quat])
        # self.arm.set_tcp_pose(target_pose)
        self.arm.move_ee(target_pose)

    def set_tcp_gripper(self, width):
        # self.arm.set_gripper_position(width)
        self.arm.move_gripper(width)

# -----------------------------------------------------------------------------
# Point Cloud Generation
# -----------------------------------------------------------------------------

def get_point_cloud(depth, intrinsics):
    # depth: (H, W)
    # intrinsics: (3, 3)
    H, W = depth.shape
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]
    
    y, x = np.indices((H, W))
    z = depth
    x = (x - cx) * z / fx
    y = (y - cy) * z / fy
    
    point_cloud = np.stack([x, y, z], axis=-1) # (H, W, 3)
    return point_cloud

def get_real_obs_dict(agent_obs, shape_meta):
    # agent_obs: {'rgb': (T, H, W, 3), 'depth': (T, H, W), 'pose': (T, D)}
    # Output: {'point_cloud': (T, N, 3 or 6), 'agent_pos': (T, D)}
    
    rgb = agent_obs['rgb']
    depth = agent_obs['depth']
    pose = agent_obs['pose']
    
    T, H, W, _ = rgb.shape
    
    # Intrinsics - HARDCODED for now, replace with actual camera intrinsics
    # Assuming 84x84 and some FOV
    intrinsics = np.array([[42.0, 0, 42.0], [0, 42.0, 42.0], [0, 0, 1]]) 
    
    point_clouds = []
    for i in range(T):
        pc = get_point_cloud(depth[i], intrinsics) # (H, W, 3)
        
        # Color
        color = rgb[i] / 255.0
        pc_color = np.concatenate([pc, color], axis=-1) # (H, W, 6)
        
        # Flatten
        pc_flat = pc_color.reshape(-1, 6)
        
        # Downsample/Crop to fixed number of points (e.g. 1024 or 512)
        # DP3 usually expects fixed number of points
        n_points = 1024 # Default for many DP3 configs
        if pc_flat.shape[0] > n_points:
            indices = np.random.choice(pc_flat.shape[0], n_points, replace=False)
            pc_flat = pc_flat[indices]
        else:
            # Pad if not enough
            indices = np.random.choice(pc_flat.shape[0], n_points, replace=True)
            pc_flat = pc_flat[indices]
            
        point_clouds.append(pc_flat)
        
    point_clouds = np.array(point_clouds) # (T, N, 6)
    
    return {
        'point_cloud': point_clouds.astype(np.float32),
        'agent_pos': pose.astype(np.float32)
    }

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

def run_single_episode(agent, policy, cfg, device, max_duration, gripper, output):
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
        obs_dict_np = get_real_obs_dict(obs, cfg.task.shape_meta)
        obs_dict = dict_apply(obs_dict_np, lambda x: torch.from_numpy(x).unsqueeze(0).to(device))
        result = policy.predict_action(obs_dict)
        del result

    ensemble_buffer = EnsembleBuffer(mode="new")
    
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
                obs_dict_np = get_real_obs_dict(obs, cfg.task.shape_meta)
                obs_dict = dict_apply(obs_dict_np, lambda x: torch.from_numpy(x).unsqueeze(0).to(device))
                
                result = policy.predict_action(obs_dict)
                raw_action = result['action'][0].detach().to('cpu').numpy()
                ensemble_buffer.add_action(raw_action, t)
            
            step_action = ensemble_buffer.get_action()
            if step_action is None:
                continue
                
            # Execute action
            # step_action: 9-dim (xyz + rot6d) or 10-dim (+gripper)
            # or 7-dim (xyz + quat) or 8-dim (+gripper)
            if agent.use_rot6d:
                agent.set_tcp_pose(step_action[:9])
                if gripper and len(step_action) > 9:
                    agent.set_tcp_gripper(step_action[9])
            else:
                agent.set_tcp_pose(step_action[:7])
                if gripper and len(step_action) > 7:
                    agent.set_tcp_gripper(step_action[7])
                
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
def main(ckpt, output, max_duration, gripper, continuous):
    global interrupted
    
    # Load Checkpoint
    payload = torch.load(open(ckpt, 'rb'), pickle_module=dill)
    cfg = payload['cfg']
    
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
    
    # Determine observation params from cfg
    try:
        agent_pos_shape = cfg.task.shape_meta.obs.agent_pos.shape
        state_dim = agent_pos_shape[0]
    except:
        state_dim = 9 # Default fallback
        
    use_rot6d = True
    model_has_gripper = False
    
    if state_dim == 7: # xyz + quat
        use_rot6d = False
        model_has_gripper = False
    elif state_dim == 8: # xyz + quat + gripper
        use_rot6d = False
        model_has_gripper = True
    elif state_dim == 9: # xyz + rot6d
        use_rot6d = True
        model_has_gripper = False
    elif state_dim == 10: # xyz + rot6d + gripper
        use_rot6d = True
        model_has_gripper = True
        
    cprint(f"Detected state dim: {state_dim}, use_rot6d: {use_rot6d}, model_has_gripper: {model_has_gripper}", "yellow")
    
    agent = DP3Agent(obs_num=n_obs_steps, gripper=model_has_gripper, use_rot6d=use_rot6d)
    
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
                
            success = run_single_episode(agent, policy, cfg, device, max_duration, gripper, episode_output)
            
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
        run_single_episode(agent, policy, cfg, device, max_duration, gripper, output)
        
    agent.arm.home_robot()
    print("Goodbye.")

if __name__ == '__main__':
    main()
