import os
import wandb
import argparse

from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from trajectory.models.general_trainer import Trainer #general trainer

from trajectory.datasets.d4rl_dataset import DiscretizedDataset
from trajectory.utils.common import set_seed
from trajectory.models.trajectory import TrajectoryModel
import torch

import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group


def ddp_setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    init_process_group(backend="gloo", rank=rank, world_size=world_size)


def create_argparser():
    parser = argparse.ArgumentParser(description="Trajectory models training hyperparameters. All can be set from command line.")
    parser.add_argument("--config", default="configs/halfcheetah_medium_gpt.yaml")
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--device", default="cpu", type=str)
    

    return parser


def run_experiment(config, seed, device):
    config.run_seed = seed
    os.makedirs(config.trainer.checkpoints_path, exist_ok=True)
    OmegaConf.save(OmegaConf.to_container(config, resolve=True), os.path.join(config.trainer.checkpoints_path, "config.yaml"))

    set_seed(seed=seed)

    trainer_conf = config.trainer
    data_conf = config.dataset

    dataset = DiscretizedDataset(
        env_name=data_conf.env_name,
        seq_len=data_conf.seq_len,
        cache_path=data_conf.cache_path,
        num_bins=data_conf.num_bins,
        discount=data_conf.discount,
        strategy=data_conf.strategy
    )
    dataloader = DataLoader(dataset, batch_size=data_conf.batch_size, shuffle=False, num_workers=8, pin_memory=True, sampler=DistributedSampler(dataset))
    model_parse = config.wandb.name.split('_')[-1]
    model = TrajectoryModel(layer_type=model_parse, **config.model)
    model = model.to(device)
    print("Device: ", device)

    model.load_state_dict(torch.load(os.path.join(config.checkpoints_path, "model_10.pt"), map_location=f"cuda:{device}"))


    model = DDP(model, device_ids=[device])

    print("Number of parameters: ")
    print(sum(p.numel() for p in model.parameters() if p.requires_grad))

    num_epochs = int(1e6 / len(dataset) * trainer_conf.num_epochs_ref)

    warmup_tokens = len(dataset) * data_conf.seq_len * config.model.transition_dim
    final_tokens = warmup_tokens * num_epochs

    wandb.init(**config.wandb, config=dict(OmegaConf.to_container(config, resolve=True)))
    
    trainer = Trainer(
        final_tokens=final_tokens,
        warmup_tokens=warmup_tokens,
        action_weight=trainer_conf.action_weight,
        value_weight=trainer_conf.value_weight,
        reward_weight=trainer_conf.reward_weight,
        learning_rate=trainer_conf.lr,
        betas=trainer_conf.betas,
        weight_decay=trainer_conf.weight_decay,
        clip_grad=trainer_conf.clip_grad,
        eval_seed=trainer_conf.eval_seed,
        eval_every=trainer_conf.eval_every,
        eval_episodes=trainer_conf.eval_episodes,
        eval_temperature=trainer_conf.eval_temperature,
        eval_discount=trainer_conf.eval_discount,
        eval_plan_every=trainer_conf.eval_plan_every,
        eval_beam_width=trainer_conf.eval_beam_width,
        eval_beam_steps=trainer_conf.eval_beam_steps,
        eval_beam_context=trainer_conf.eval_beam_context,
        eval_sample_expand=trainer_conf.eval_sample_expand,
        eval_k_obs=trainer_conf.eval_k_obs,  # as in original implementation
        eval_k_reward=trainer_conf.eval_k_reward,
        eval_k_act=trainer_conf.eval_k_act,
        checkpoints_path=trainer_conf.checkpoints_path,
        save_every=1,
        device=device
    )
    trainer.train(
        model=model,
        dataloader=dataloader,
        num_epochs=num_epochs
    )


def main(rank: int, world_size: int):

    ddp_setup(rank, world_size)
    args, override = create_argparser().parse_known_args()
    config = OmegaConf.merge(
        OmegaConf.load(args.config),
        OmegaConf.from_cli(override)
    )
    run_experiment(
        config=config,
        seed=args.seed,
        device=rank #args.device
    )

    print(f'Config: {config}')

    destroy_process_group()


if __name__ == "__main__":
    
    world_size = torch.cuda.device_count()
    mp.spawn(main, args=[world_size], nprocs=world_size)
