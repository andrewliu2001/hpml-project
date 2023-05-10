# HPML COMS6998 Project: #

Collaborators: Andrew Liu, Kristen Surrao

## Description: ##
In this class project, we experiment with the novel Hyena continuous convolution kernel as an alternative to transformers for efficiently capturing long-range dependencies in offline reinforcement learning. Specifically, we incorporated the Hyena module into the Trajectory Model, performed hyperparameter sweeps and profiling, and performed optimizations including distributed training and post-training static quantization. For simplicity, we only used the halfcheetah-medium-v2 environment on Gym/MuJOCO.

## Outline of repo: ##
train.py calls a trainer object, which is under trajectory/models/general_trainer.py. You can find the boilerplate Pytorch training loop in general_trainer.py. general_trainer then uses a generic TrajectoryModel object (a wrapper around either Hyena layers or GPT layers), which is written in trajectory.py. The trajectory/planning folder contains utilities for beam search, whereas trajectory/utils contains other utilities such as learning rate scheduling and trajectory discretization. Note that much of the codebase is based on https://github.com/Howuhh/faster-trajectory-transformer (see acknowledgements section)


## Installing dependencies: ##

## Scripts: ##
Training:
```
python train.py --config="configs/medium/halfcheetah_medium_hyena.yaml" --device="cuda" --mem=120G
```

## Evaluation/profiling: ##
```
python eval.py --config="configs/eval_base.yaml" --device="cuda" --seed="42" checkpoints_path="checkpoints/halfcheetah-medium-v2-hyena/uniform/baseline" beam_context=5 beam_steps=5 beam_width=32

python profile.py --config="configs/eval_base.yaml" --device="cuda" --seed="42" checkpoints_path="checkpoints/halfcheetah-medium-v2-hyena/uniform/baseline" beam_context=5 beam_steps=5 beam_width=32
```

## Results: ##
### W&B sweep: ###

Note that during these sweeps, we kept Hyena’s convolution window the same size as the attention window in GPT-based Trajectory Model (len=250). This is so we can maintain same batch size of 256 when training on a single GPU. When training the best model, we increased the length to max length and decreased batch size to 64 to fit into memory, resulting in a much longer time-per epoch (see side-by-side comparison of Hyena and GPT).

To save time, each Hyena sweep lasted only 2 epochs. When using a window size of 250, this training duration proved sufficient for the model to converge. The sweeps suggest that, with a small fixed convolution window, model size is clearly the dominant performance factor. After fixing the model to 11-12 layers, we see that embedding_dropout becomes the second most important hyperparameter. 

![Sweep curve](https://github.com/andrewliu2001/hpml-project/blob/tuning/assets/sweep.png)
![Sweep table](https://github.com/andrewliu2001/hpml-project/blob/tuning/assets/sweep_table.png)
![Importances](https://github.com/andrewliu2001/hpml-project/blob/tuning/assets/importance.png)

### Side-by-side comparison of Hyena and GPT-based Trajectory Models: ###
For the comparison against GPT, we increased the Hyena window length to 1000 and used the maximal possible batch size in each setting. We also performed distributed training on the Hyena models and observe a significant speedup. We note that a Hyena model with 800K parameters is able to perform competitively against the 1.5M-parameter Hyena models and the GPT-based architecture, suggesting the effectiveness of parameter-saving using large, continuous convolution kernels. Note that we can further improve performance dramatically by incorporating FlashConvolution - an optimization for computing faster FFTs. However, due to version constraints on Habanero GPUs were were unable to incorporate this before the deadline. 

![Hyena vs GPT]([https://github.com/andrewliu2001/hpml-project/blob/tuning/assets/importance.png](https://github.com/andrewliu2001/hpml-project/blob/tuning/assets/hyenavsgpt.png)


### Profiling Hyena at inference time on CPU: ###
Compute-wise, cost is dominated by FFT/IFFT and linear layers, which is expected since Hyena uses linear layers to generate its continuous convolution kernel (which is then computed using FFT). Speeding up linear layers during inference time can be done using static quantization (quantizing both weight and activations post-training as opposed to just weights). Since we are interested in large kernels (of length >=1000), it does not make sense to experiment with other convolution algorithms, so we decided to stick with FFT. 

![Hyena profile](https://github.com/andrewliu2001/hpml-project/blob/tuning/assets/hyena_profile.png)

### Quantization: ###
Note that we can only dynamically quantize linear layers. The short kernel 1-D convolution is not dynamically quantizable. We observe an overall 27% speedup, linear layers sped up by 25%, and subsequent FFT sped up by 50%. Moreoever, we were able to recover on average 94.3% of original rewards after quantization.

![Hyena dynamic quantization](https://github.com/andrewliu2001/hpml-project/blob/tuning/assets/quantization.png)



###Acknowledgements:
1. Codebase forked from https://github.com/Howuhh/faster-trajectory-transformer
2. Hyena module from https://github.com/HazyResearch/safari
3. Sashimi/S4 from https://github.com/HazyResearch/state-spaces
