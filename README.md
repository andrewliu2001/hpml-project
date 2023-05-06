# HPML COMS6998 Project

Collaborators: Andrew Liu, Kristen Surrao

Description:
In this class project, we experiment with the novel Hyena continuous convolution kernel as an alternative to transformers for efficiently capturing long-range dependencies in offline reinforcement learning. Specifically, we incorporated the Hyena module into the Trajectory Model, performed hyperparameter sweeps and profiling, and performed optimizations including distributed training and post-training static quantization. For simplicity, we only used

Outline of repo:
1. train.py calls a trainer object, which is under trajectory/models/general_trainer.py. You can find the boilerplate Pytorch training loop in general_trainer.py. general_trainer then uses a generic TrajectoryModel object (a wrapper around either Hyena layers or GPT layers), which is written in trajectory.py. 
2. 

Installing dependencies:

Scripts:
```
python train.py --config="configs/medium/halfcheetah_medium_hyena.yaml" --device="cuda" --mem=120G
```


Results:



Acknowledgements:
1. Codebase forked from https://github.com/Howuhh/faster-trajectory-transformer
2. Hyena module from https://github.com/HazyResearch/safari
3. Sashimi/S4 from https://github.com/HazyResearch/state-spaces
