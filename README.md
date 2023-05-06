# HPML COMS6998 Project:

Collaborators: Andrew Liu, Kristen Surrao

Description:
In this class project, we experiment with the novel Hyena continuous convolution kernel as an alternative to transformers for efficiently capturing long-range dependencies in offline reinforcement learning. Specifically, we incorporated the Hyena module into the Trajectory Model, performed hyperparameter sweeps and profiling, and performed optimizations including distributed training and post-training static quantization. For simplicity, we only used

Outline of repo:
1. train.py calls a trainer object, which is under trajectory/models/general_trainer.py. You can find the boilerplate Pytorch training loop in general_trainer.py. general_trainer then uses a generic TrajectoryModel object (a wrapper around either Hyena layers or GPT layers), which is written in trajectory.py. 
2. The planning folder

Installing dependencies:

Scripts:
Training:
```
python train.py --config="configs/medium/halfcheetah_medium_hyena.yaml" --device="cuda" --mem=120G
```

Evaluation/profiling:
```
python eval.py --config="configs/eval_base.yaml" --device="cuda" --seed="42" checkpoints_path="checkpoints/halfcheetah-medium-v2-hyena/uniform/baseline" beam_context=5 beam_steps=5 beam_width=32

python profile.py --config="configs/eval_base.yaml" --device="cuda" --seed="42" checkpoints_path="checkpoints/halfcheetah-medium-v2-hyena/uniform/baseline" beam_context=5 beam_steps=5 beam_width=32
```

Results:
W&B sweep:
![Sweep curve](https://github.com/andrewliu2001/hpml-project/blob/tuning/assets/sweep.png)
![Sweep table](https://github.com/andrewliu2001/hpml-project/blob/tuning/assets/sweep_table.png)
![Importances](https://github.com/andrewliu2001/hpml-project/blob/tuning/assets/importance.png)

Profiling Hyena at inference time on CPU:
![Hyena profile](https://github.com/andrewliu2001/hpml-project/blob/tuning/assets/hyena_profile.png)

Acknowledgements:
1. Codebase forked from https://github.com/Howuhh/faster-trajectory-transformer
2. Hyena module from https://github.com/HazyResearch/safari
3. Sashimi/S4 from https://github.com/HazyResearch/state-spaces
