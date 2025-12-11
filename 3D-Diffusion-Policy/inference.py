if __name__ == "__main__":
    import sys
    import os
    import pathlib

    ROOT_DIR = str(pathlib.Path(__file__).parent.parent.parent)
    sys.path.append(ROOT_DIR)
    os.chdir(ROOT_DIR)

import json
import os
import hydra
import torch
import dill
from omegaconf import OmegaConf
import pathlib
from train import TrainDP3Workspace
from typing import Dict, Callable, List
import numpy as np

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

OmegaConf.register_new_resolver("eval", eval, replace=True)
    
def take_an_obs(batch):
    obs = batch['obs']
    obs = dict_apply(obs, lambda x: np.array(x))
    
    return obs


@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.joinpath(
        'diffusion_policy_3d', 'config'))
)
def main(cfg):
    workspace = TrainDP3Workspace(cfg)

    obs = {
        'agent_pos': None,
        'point_cloud': None,
    }

    # take_an_obs()

    with open('infer/infer.json', 'r', encoding='utf-8') as f:
        obs = take_an_obs(json.load(f))

    action = workspace.inference(obs)

    with open('infer/action.json', 'w', encoding='utf-8') as f:
        json.dump(action.tolist(), f, indent='\t')
    

if __name__ == "__main__":
    main()
