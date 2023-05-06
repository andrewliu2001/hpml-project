import torch
from torch.profiler import profile, record_function, ProfilerActivity
from trajectory.models.trajectory import TrajectoryModel
from eval import create_argparser
from omegaconf import OmegaConf
from trajectory.utils.common import set_seed
import os

"""
Run:
python profile.py --config="configs/eval_base.yaml" --device="cuda" --seed="42" checkpoints_path="checkpoints/halfcheetah-medium-v2-hyena/uniform/baseline" beam_context=5 beam_steps=5 beam_width=32

"""


def model_profile(config, seed, device):
    """
        model: nn.Module
    """
    set_seed(seed=seed)
    run_config = OmegaConf.load(os.path.join(config.checkpoints_path, "config.yaml"))
    discretizer = torch.load(os.path.join(config.checkpoints_path, "discretizer.pt"), map_location=device)

    model_parse='hyena'
    model = TrajectoryModel(layer_type=model_parse, **run_config.model)
    model.eval()
    model.to(device)
    model.load_state_dict(torch.load(os.path.join(config.checkpoints_path, config.model_name), map_location=device))


    example_context = torch.ones((1600,1)).int().to(device)
    example_state = None
    example_out, example_next_state = model(example_context, example_state)

    with profile(activities=[ProfilerActivity.CPU], profile_memory=True, record_shapes=True) as prof:
        with record_function("model_inference"):
            example_out, example_next_state = model(example_context, example_next_state)


    print(f"The model and data are on: {device}")

    print("Sort by self_cuda_time_total:")
    print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=10))
    print("Sort by cuda_time_total:")
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
    print("Sort by cuda_memory_usage:")
    print(prof.key_averages().table(sort_by="cuda_memory_usage", row_limit=10))

    print("----------------------------------------------------------------------------")

    print("Sort by self_cpu_time_total:")
    print(prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=10))
    print("Sort by cpu_time_total:")
    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))
    print("Sort by cpu_memory_usage:")
    print(prof.key_averages().table(sort_by="cpu_memory_usage", row_limit=10))

    prof.export_chrome_trace(model_parse+"_"+device+"_trace.json")


def main():
    args, override = create_argparser().parse_known_args()
    config = OmegaConf.merge(
        OmegaConf.load(args.config),
        OmegaConf.from_cli(override)
    )

    model_profile(config=config, seed=args.seed, device='cpu')#args.device)
    
    


if __name__ == "__main__":
    main()
