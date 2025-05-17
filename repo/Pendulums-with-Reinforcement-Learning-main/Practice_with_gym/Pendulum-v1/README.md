## Pendulum-v1 with DDPG

Here, another classic control problem, pendulum-v1, is solved with *DDPG* (Deep Deterministic Policy Gradient). It seems that DDPG tends to have large fluctuations w.r.t. episodic rewards, and does not converge well; also, it is very sensitive to hyperparameters, thus hyperparameter tuning was an important process.

### Hyperparameters
After a few trials-and-errors, the following hyperpameters seem to give promising results.

```python
self.alpha_actor = 1e-4
self.alpha_critic = 1e-3
self.gamma = 0.95
self.tau = 1e-3

buffer_max_length = 20000
batch_size = 32
learning_starts = 1000

mu = 0
theta = 0.15
sigma = 0.2
```

### Network Structure
Critic: 4 -> 32 -> 32 -> 16 -> 1 (LeakyReLU)
Actor: 3 -> 64 -> 32 -> 16 -> 1 (LeakyReLU)

### Remarks
It was observed that normalization of any kind did not benefit the learning process at all, including both LayerNorm and BatchNorm and rather harms the learning. This may have been due to errors in my implementation of the algorithm. However, due to the shortcomings of the algorithm as explained above, I've decided to move on to *SAC* for more complex environments.
