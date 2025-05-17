## Double Inverted Pendulum (Stabilization)

Here, another classic control environment, double inverted pendulum (DIP) is solved with *SAC* (Soft Actor-Critic), an effective off-policy reinforcement learning algorithm. About 600 episodes are enough for the agent to last for hours.

### Environment

A custom environment was made, which was easily done by tweaking the cartpole environment from gym. The custom environment is under the folder **user_env_gym**, and the folder needs to be placed in the same directory as the agent.py and Model.py files. Note that one could easily swtich-up to the double inverted pendulum environment from Mujoco, which utilizes a cart to stabilize the pendulum. Just change the state dimension, etc. In our environment, rather, a torque on the first pendulum is applied to control it. Our agent can readily solve both environments. 

### Hyperparameters
```python
self.GAMMA = 0.99
self.BATCH_SIZE = 128
self.BUFFER_SIZE = 1_000_000
self.ACTOR_LEARNING_RATE = 3e-4
self.CRITIC_LEARNING_RATE = 3e-4
self.TAU = 5e-3
self.ALPHA = 1/3

```

### Remarks
In the mujoco environment, 11 state inputs are given. Only 6 is used in our environment. We noticed that the agent learns way better when the inputs themselves are nonlinear; that is, instead of giving $\theta_{1}$ and $\theta_{2}$, it is better off giving $\sin(\theta_{1})$ and $\sin(\theta_{2})$.


### Reference
*수학으로 풀어보는 강화학습 원리와 알고리즘 개정판 - 박성수 지음*
