import os
import torch
import argparse
import numpy as np

from tqdm.auto import trange
from omegaconf import OmegaConf

from stable_baselines3.common.vec_env import DummyVecEnv

from trajectory.models.trajectory import TrajectoryModel
from trajectory.utils.common import set_seed
from trajectory.utils.env import create_env, rollout, vec_rollout

from torch.profiler import profile, record_function, ProfilerActivity

def create_argparser():
    parser = argparse.ArgumentParser(description="Trajectory Transformer evaluation hyperparameters. All can be set from command line.")
    parser.add_argument("--config", default="configs/medium/halfcheetah_medium.yaml")
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--device", default="cpu", type=str)

    return parser


def run_experiment(config, seed, device):
    set_seed(seed=seed)

    run_config = OmegaConf.load(os.path.join(config.checkpoints_path, "config.yaml"))
    discretizer = torch.load(os.path.join(config.checkpoints_path, "discretizer.pt"), map_location=device)


    model_parse='gpt'
    
    model = TrajectoryModel(layer_type=model_parse, **run_config.model)
    model.eval()
    model.to(device)
    model.load_state_dict(torch.load(os.path.join(config.checkpoints_path, config.model_name), map_location=device))
    
    tokens = torch.ones((100,1), dtype=torch.long)
    tokens.to(device)
    model(tokens)




def main():
    args, override = create_argparser().parse_known_args()
    config = OmegaConf.merge(
        OmegaConf.load(args.config),
        OmegaConf.from_cli(override)
    )
    #with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], use_cuda=True) as prof:
        #with record_function("run_experiment"):
    run_experiment(config=config, seed=args.seed, device=args.device)
    #print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))
    #prof.export_chrome_trace("trace_gpt_vectorizedrollouts_attentioncaching.json")
    #print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))


if __name__ == "__main__":
    main()
