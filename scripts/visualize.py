import visualizer
import json

from typing import Dict, Callable, List
import torch
import numpy as np

# your_pointcloud = ... # your point cloud data, numpy array with shape (N, 3) or (N, 6)
def dict_apply(
        x: Dict[str, torch.Tensor], 
        func: Callable[[torch.Tensor], torch.Tensor]
        ) -> Dict[str, torch.Tensor]:
    result = dict()
    for key, value in x.items():
        if isinstance(value, dict):
            result[key] = dict_apply(value, func)
        else:
            result[key] = func(value)
    return result

def take_an_obs(batch):
    obs = batch['obs']
    obs = dict_apply(obs, lambda x: np.array(x))
    
    return obs


with open('tmp.json', 'r', encoding='utf-8') as f:
        pointcloud = json.load(f)
        pointcloud = np.array(pointcloud)

print(pointcloud.shape)

visualizer.visualize_pointcloud(pointcloud)