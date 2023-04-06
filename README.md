# Sparse-attention: Efficient sequence modeling project

To run training script:

```python
python train.py --config="configs/medium/<env>_<difficulty>_<layer_type>.yaml" --device="cuda" --mem=120G
```


Core implementations:
1. Hyena architecture + profiling + WANDB tuning + comparison of results on D4RL against trajectory transformer (GPT-based). --> Finish by April 22.
2. Perform Pykeops optimizations.


Secondary ideas:
1. Incorporate FlashAttention and FlashConv and measure performance speedups.
2. Implement MCTS planning.

Other ideas:
1. Online PCA vs S4/Sashimi. --> Can we get rid of complicated initialization schemes and view the entire LRD problem as a dimensionality reduction problem in kernel space?
2. RNN region proposal for transformer processing vs Reformer, Memorizing Transformer, and Block Recurrent Transformer.
3. Vanishing gradient due to O(N) bottleneck in RNNs:
    a. Review GNN literature and answer the question: Is the gradient signal conserved? How much of a backprop update do the respective time steps get?
4. Hyperbolic embeddings
5. Maximum entropy offline RL? Review principle of maximum entropy in variational inference.


Acknowledgements:
1. Structure follows from https://github.com/Howuhh/faster-trajectory-transformer
2. Hyena from https://github.com/HazyResearch/safari
3. Sashimi/S4 from https://github.com/HazyResearch/state-spaces
