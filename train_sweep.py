import os
import wandb
import argparse

from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from trajectory.models.general_trainer import Trainer #general trainer

from trajectory.datasets.d4rl_dataset import DiscretizedDataset
from trajectory.utils.common import set_seed
from trajectory.models.trajectory import TrajectoryModel

def create_argparser():
    parser = argparse.ArgumentParser(description="Trajectory models training hyperparameters. All can be set from command line.")
    parser.add_argument("--config", default="configs/halfcheetah_medium_gpt.yaml")
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--device", default="cpu", type=str)
    

    return parser

def build_sweep_config():
    '''
    hyperparameter sweep method. Initially use random to get general idea of 
    good hyperparameters. Once we narrow them down, we can change to a Bayesian approach if necessary
    '''
    sweep_config = {
        'method': 'random'
    }

    #goal of hyperparameter sweep
    metric = {
        'name': 'loss',
        'goal': 'minimize'   
    }
    sweep_config['metric'] = metric

    #parameters for hyperparameter sweep
    parameters_dict = {
        'num_layers': {
            'values': [11, 12]
        },
        'embedding_dropout':{
            'distribution': 'uniform',
            'min': 0.0,
            'max': 0.3,
        },
        'residual_dropout': {
            'distribution': 'uniform',
            'min': 0.0,
            'max': 0.3
        },
        'attention_dropout': {
            'distribution': 'uniform',
            'min': 0.0,
            'max': 0.3
        }
    }

    sweep_config['parameters'] = parameters_dict

    return sweep_config



def run_experiment():

    seed = args.seed
    device = args.device

    wandb.init(project=config.wandb.name)

    config.model.update(wandb.config)

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
    dataloader = DataLoader(dataset, batch_size=data_conf.batch_size, shuffle=True, num_workers=8, pin_memory=True)
    model_parse = config.wandb.name.split('_')[-1]

    model = TrajectoryModel(layer_type=model_parse, **config.model)
    model.to(device)
    num_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("number of trainable parameters is ", num_trainable_params)
    wandb.log({'trainable_params': num_trainable_params})

    num_epochs = int(1e6 / len(dataset) * trainer_conf.num_epochs_ref)
    print("number of epochs is ", num_epochs)

    warmup_tokens = len(dataset) * data_conf.seq_len * config.model.transition_dim
    final_tokens = warmup_tokens * num_epochs

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


if __name__ == "__main__": #run full sweep

    args, override = create_argparser().parse_known_args()
    config = OmegaConf.merge(
        OmegaConf.load(args.config),
        OmegaConf.from_cli(override)
    )
    
    sweep_config = build_sweep_config()
    sweep_id = wandb.sweep(sweep=sweep_config, project=config.wandb.name)
    #sweep_id = 'kmsurrao/halfcheetah_medium_hyena/v51lsrud' #remove and uncomment above
    wandb.agent(sweep_id, function=run_experiment, count=10)

    print(f'Device: {args.device}')
    print(f'Config: {config}')
