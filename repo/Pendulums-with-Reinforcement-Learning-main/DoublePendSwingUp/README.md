# Double Inverted Pendulum Swing-Up & Stabilization

Here, the same custom environment is used, and the double inverted pendulum initially swings-up to make the pendulum stand upright and then stabilizes it. We realized that this would be a more challenging task to solve than the simple stabilization task. Indeed, much more training time was necessary, but after about 5000 episodes, the agent seems to be capable of nailing the task. A succesfully trained agent is saved under the folder **saved_models/08:34/**.

## Environment

The same custom environment was used. A rough reward function was used considering only the position of the pendulum; when they are in the appropriate boundary (in terms of angles), a reward of +1 was given, and -1 for all other was given. The episode is truncated after 500 time steps, so minimum reward is -500. When the environment is reset, the pendulum may take any position, with some initial angular speed (-0.2~0.2 rad/s). I believe this makes the task more challenging for our agent, yet enhances its generalization ability.

## Hyperparameters

```python
self.GAMMA = 0.99
self.BATCH_SIZE = 256
self.BUFFER_SIZE = 1_000_000
self.ACTOR_LEARNING_RATE = 3e-4
self.CRITIC_LEARNING_RATE = 3e-4
self.TAU = 5e-3
self.ALPHA = 1/2
```
