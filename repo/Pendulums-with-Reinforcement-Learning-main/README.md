# Pendulums with RL

## Various Pendulum Tasks

Here, various environments that deal with stabilizing or swinging-up pendulum(s) are solved with **RL** (Reinforcement Learning) approaches. Each folders
contain different *README.md* files, which you can read to find more about each subtask.

## Environments solved

|               Task               |     Env     | Algorithm |
| :------------------------------: | :---------: | :-------: |
|  Single Pendulum Stabilization   | Cartpole-v1 |   DDPG    |
|     Single Pendulum Swing-Up     | Pendulum-v1 |   DDPG    |
|     Double Pendulum Swing-Up     | Acrobot-v1  |    SAC    |
|  Double Pendulum Stabilization*  | custom env  |    SAC    |
| Double Pendulum Swing-Up & Stab. | custom env  |    SAC    |

*The double pendulum stabilization implementation was also succesfully tested on the mujoco environment!

Note: The first three environments are from the gym pacakge. They are all under the folder Practice_with_gym.

## Algorithm Used

* **DDPG**
* **SAC** (seems way better, at least for pendulum-related tasks)

## Reference & Remarks

* [Pytorch Implementation of SAC](https://github.com/RoyElkabetz/SAC_with_PyTorch)
  * This one shows how to implement the older version of *SAC* with *PyTorch*. A value network is included in the network structure, which can still be used,
  but seems like you can get rid of it. From this code, I realized that using .rsample() and .sample() in the right places is something that one should be mindful of when implementing SAC.
* Book: **수학으로 풀어보는 강화학습 원리와 알리즘**, Author: *박성수*  [Personal Blog Link](https://pasus.tistory.com/)
  * Also referred to this book to study how *SAC* and *DDPG* works! The example code was in tensorflow2, so had to tweak a lot of things myself. Also, the example code (which was for pendulum-v1) could not solve more complex environments, so had to figure out what the problem was. Some hyperparameter tuning, and the reparameterization trick (see above) was the deal.

## NOTE

This repository was created just for myself to study reinforcement learning algorithms. No any other purposes, and I looked up many other open-source codes.
See the reference section above (or my code) if you also need some help implementing!
