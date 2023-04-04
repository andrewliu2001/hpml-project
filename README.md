#Sparse-attention: Efficient sequence modeling project

Core implementations:
1. Hyena architecture + profiling + WANDB tuning + comparison of results on D4RL against Decision/trajectory transformer (GPT-based). --> Finish by April 22.

Other ideas:
1. Online PCA vs S4/Sashimi. --> Can we get rid of complicated initialization schemes and view the entire LRD problem as a dimensionality reduction problem in kernel space?
2. RNN region proposal for transformer processing vs Reformer the efficient transformer. Understand how LSH works in Reformer.
3. Vanishing gradient due to O(N) bottleneck in RNNs:
    a. Review GNN literature and answer the question: Is the gradient signal conserved? How much of a backprop update do the respective time steps get?
4. Hyperbolic embeddings
5. Maximum entropy offline RL? Review principle of maximum entropy in variational inference.