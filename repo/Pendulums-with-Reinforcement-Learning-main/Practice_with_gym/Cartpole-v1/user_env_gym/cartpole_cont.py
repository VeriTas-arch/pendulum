"""
File Name : cartpole_cont.py
Similar with official gymnasium cartpole, but has continuous action space
Consider friction between cart&floor, cart&pole
"""


import math
from typing import Optional, Union

import gymnasium as gym
import numpy as np
import scipy.stats
from gymnasium import logger, spaces
from gymnasium.envs.classic_control import utils
from gymnasium.error import DependencyNotInstalled


class CartPoleEnv(gym.Env[np.ndarray, Union[int, np.ndarray]]):
    """
    ### Action Space
    The action is a `ndarray` with shape `(1,)` which can take values `{-20, 20}` for default.
    Action value corresponds to the force pushing or pulling cart
    | Num | Action                 |
    |-----|------------------------|
    | 0   | Amount of force        |

    ### Observation Space
    The observation is a `ndarray` with shape `(4,)` with the values corresponding to the following positions and velocities:
    | Num | Observation           | Min                 | Max               |
    |-----|-----------------------|---------------------|-------------------|
    | 0   | Cart Position         | -4.8                | 4.8               |
    | 1   | Cart Velocity         | -Inf                | Inf               |
    | 2   | Pole Angle            | ~ -0.418 rad (-24°) | ~ 0.418 rad (24°) |
    | 3   | Pole Angular Velocity | -Inf                | Inf               |
    **Note:** While the ranges above denote the possible values for observation space of each element,
        it is not reflective of the allowed values of the state space in an unterminated episode. Particularly:
    -  The cart x-position (index 0) can be take values between `(-4.8, 4.8)`, but the episode terminates
       if the cart leaves the `(-2.4, 2.4)` range.
    -  The pole angle can be observed between  `(-.418, .418)` radians (or **±24°**), but the episode terminates
       if the pole angle is not in the range `(-.2095, .2095)` (or **±12°**)

    ### Rewards
    There are various modes of rewards implemented, and you can choose one of them when initialize.
    1) Constant reward (default) (reward_mode = 0)
        Since the goal is to keep the pole upright for as long as possible, a reward of `+1` for every step taken,
        including the termination step, is allotted. The threshold for rewards is 475 for v1.
    2) Discrete reward (reward_mode = 1)
        Gives reward = 2 when position is between -1 and 1. Else, gives reward = 1
    3) Continuous reward (reward_mode = 2)
        Gives reward according to normal distribution. (Time consuming)

    ### Starting State
    All observations are assigned a uniformly random value in `(-0.05, 0.05)`

    ### Episode End
    The episode ends if any one of the following occurs:
    1. Termination: Pole Angle is greater than ±12°
    2. Termination: Cart Position is greater than ±2.4 (center of the cart reaches the edge of the display)
    3. Truncation: Episode length is greater than 500 (200 for v0)
    """

    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 50,
    }

    def __init__(self, render_mode: Optional[str] = None, reward_mode: int = 0):
        assert reward_mode == 0 or reward_mode == 1 or reward_mode == 2, "Wrong reward mode parameter"

        self.gravity = 9.8
        self.masscart = 1.0
        self.masspole = 0.1
        self.total_mass = self.masspole + self.masscart
        self.length = 0.5  # actually half the pole's length
        self.polemass_length = self.masspole * self.length * 2
        self.fric_coef = 0.05  # friction between floor and cart
        self.fric_rot = 0.03  # friction between cart and pole
        self.force_mag = 20.0  # max. abs. value of force. (-force_mag < force < +force_mag)
        self.tau = 0.02  # seconds between state updates
        self.kinematics_integrator = "euler"  # simulation step update mode. supports euler and semi-implicit euler

        self.destination = 0.0  # destination x coordinate (only for modified ISE reward)

        # Angle at which to fail the episode
        self.theta_threshold_radians = 12 * 2 * math.pi / 360
        self.x_threshold = 2.4

        # Angle limit set to 2 * theta_threshold_radians so failing observation
        # is still within bounds.
        high = np.array(
            [
                self.x_threshold * 2,
                np.finfo(np.float32).max,  # means 'actually infinite'
                self.theta_threshold_radians * 2,
                np.finfo(np.float32).max,  # means 'actually infinite'
            ],
            dtype=np.float32,
        )

        # Define action and observation space. spaces.Discrete() for discrete action/observation space
        self.action_space = spaces.Box(-self.force_mag, self.force_mag, dtype=np.float32)
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)

        # States history
        self.states = []

        self.render_mode = render_mode
        self.reward_mode = reward_mode

        self.screen_width = 600
        self.screen_height = 400
        self.screen = None
        self.clock = None
        self.isopen = True
        self.state = None

        self.steps_beyond_terminated = None

    def step(self, action):
        err_msg = "invalid action detected. Please set action value between {0} and {1}".format(-self.force_mag, self.force_mag)
        ### action을 torch.tensor로 받아서 밑부분을 수정했습니다
        import torch
        if isinstance(action, torch.Tensor): action = action.item()
        assert -self.force_mag <= action and action <= self.force_mag, err_msg
        assert self.state is not None, "Call reset before using step method."
        x, x_dot, theta, theta_dot = self.state
        force = action
        costheta = math.cos(theta)
        sintheta = math.sin(theta)

        # For the interested reader:
        # https://coneural.org/florian/papers/05_cart_pole.pdf

        # without friction

        # temp = (
        #     force + self.polemass_length * theta_dot**2 * sintheta
        # ) / self.total_mass
        # thetaacc = (self.gravity * sintheta - costheta * temp) / (
        #     self.length * (4.0 / 3.0 - self.masspole * costheta**2 / self.total_mass)
        # )
        # xacc = temp - self.polemass_length * thetaacc * costheta / self.total_mass

        # with friction

        thetaacc = 0 # for initial n_c calculation
        n_c = self.total_mass * self.gravity - self.polemass_length * (thetaacc * sintheta + theta_dot**2 * costheta)
        temp = (
            force + self.polemass_length * theta_dot**2 * (sintheta + self.fric_coef * math.copysign(1, n_c * x_dot) * costheta)
        ) / self.total_mass
        temp2 = (
            force + self.polemass_length * (theta_dot**2 * sintheta - thetaacc * costheta)
        ) / self.total_mass
        thetaacc = (self.gravity * sintheta - costheta * (temp - self.fric_coef * self.gravity * math.copysign(1, n_c * x_dot)) - (self.fric_rot * theta_dot) / (self.polemass_length)) / (
            self.length * (4.0 / 3.0 - self.masspole * costheta * (costheta - self.fric_coef * math.copysign(1, n_c * x_dot)) / self.total_mass)
        )
        xacc = temp2 - self.fric_coef * n_c * math.copysign(1, n_c * x_dot) / self.total_mass

        

        if self.kinematics_integrator == "euler":
            x = x + self.tau * x_dot
            x_dot = x_dot + self.tau * xacc
            theta = theta + self.tau * theta_dot
            theta_dot = theta_dot + self.tau * thetaacc
        else:  # semi-implicit euler
            x_dot = x_dot + self.tau * xacc
            x = x + self.tau * x_dot
            theta_dot = theta_dot + self.tau * thetaacc
            theta = theta + self.tau * theta_dot

        self.states.append(self.state)
        self.state = (x, x_dot, theta, theta_dot)

        terminated = bool(
            x < -self.x_threshold
            or x > self.x_threshold
            or theta < -self.theta_threshold_radians
            or theta > self.theta_threshold_radians
        )

        if terminated:
            self.states.append(self.state)

        if not terminated:
            reward = 0.0
            if self.reward_mode == 0:
                reward = 1.0
            if self.reward_mode == 1:
                if self.state[0] <= 1 and self.state[0] >= -1:
                    reward = 2.0
                else:
                    reward = 1.0
            if self.reward_mode == 2:
                norm_dist_x = scipy.stats.norm(loc = 0, scale = 0.5)
                norm_dist_theta = scipy.stats.norm(loc = 0, scale = 0.1)
                reward = 0.5 * math.sqrt(2 * 3.14) * norm_dist_x.pdf(self.state[0]) + 0.1 * math.sqrt(2 * 3.14) * norm_dist_theta.pdf(self.state[2])
        elif self.steps_beyond_terminated is None:
            # Pole just fell!
            self.steps_beyond_terminated = 0
            reward = 1.0
        else:
            if self.steps_beyond_terminated == 0:
                logger.warn(
                    "You are calling 'step()' even though this "
                    "environment has already returned terminated = True. You "
                    "should always call 'reset()' once you receive 'terminated = "
                    "True' -- any further steps are undefined behavior."
                )
            self.steps_beyond_terminated += 1
            reward = 0.0

        if self.render_mode == "human":
            self.render()
        return np.array(self.state, dtype=np.float32), reward, terminated, False, {}

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ):
        super().reset(seed=seed)
        # Note that if you use custom reset bounds, it may lead to out-of-bound
        # state/observations.
        low, high = utils.maybe_parse_reset_bounds(
            options, -0.05, 0.05  # default low
        )  # default high
        self.state = self.np_random.uniform(low=low, high=high, size=(4,))
        self.states = []
        self.steps_beyond_terminated = None

        if self.render_mode == "human":
            self.render()
        return np.array(self.state, dtype=np.float32), {}

    def render(self):
        if self.render_mode is None:
            gym.logger.warn(
                "You are calling render method without specifying any render mode. "
                "You can specify the render_mode at initialization, "
                f'e.g. gym("{self.spec.id}", render_mode="rgb_array")'
            )
            return

        try:
            import pygame
            from pygame import gfxdraw
        except ImportError:
            raise DependencyNotInstalled(
                "pygame is not installed, run `pip install gym[classic_control]`"
            )

        if self.screen is None:
            pygame.init()
            if self.render_mode == "human":
                pygame.display.init()
                self.screen = pygame.display.set_mode(
                    (self.screen_width, self.screen_height)
                )
            else:  # mode == "rgb_array"
                self.screen = pygame.Surface((self.screen_width, self.screen_height))
        if self.clock is None:
            self.clock = pygame.time.Clock()

        world_width = self.x_threshold * 2
        scale = self.screen_width / world_width
        polewidth = 10.0
        polelen = scale * (2 * self.length)
        cartwidth = 50.0
        cartheight = 30.0

        if self.state is None:
            return None

        x = self.state

        self.surf = pygame.Surface((self.screen_width, self.screen_height))
        self.surf.fill((255, 255, 255))

        l, r, t, b = -cartwidth / 2, cartwidth / 2, cartheight / 2, -cartheight / 2
        axleoffset = cartheight / 4.0
        cartx = x[0] * scale + self.screen_width / 2.0  # MIDDLE OF CART
        carty = 100  # TOP OF CART
        cart_coords = [(l, b), (l, t), (r, t), (r, b)]
        cart_coords = [(c[0] + cartx, c[1] + carty) for c in cart_coords]
        gfxdraw.aapolygon(self.surf, cart_coords, (0, 0, 0))
        gfxdraw.filled_polygon(self.surf, cart_coords, (0, 0, 0))

        l, r, t, b = (
            -polewidth / 2,
            polewidth / 2,
            polelen - polewidth / 2,
            -polewidth / 2,
        )

        pole_coords = []
        for coord in [(l, b), (l, t), (r, t), (r, b)]:
            coord = pygame.math.Vector2(coord).rotate_rad(-x[2])
            coord = (coord[0] + cartx, coord[1] + carty + axleoffset)
            pole_coords.append(coord)
        gfxdraw.aapolygon(self.surf, pole_coords, (202, 152, 101))
        gfxdraw.filled_polygon(self.surf, pole_coords, (202, 152, 101))

        gfxdraw.aacircle(
            self.surf,
            int(cartx),
            int(carty + axleoffset),
            int(polewidth / 2),
            (129, 132, 203),
        )
        gfxdraw.filled_circle(
            self.surf,
            int(cartx),
            int(carty + axleoffset),
            int(polewidth / 2),
            (129, 132, 203),
        )

        gfxdraw.hline(self.surf, 0, self.screen_width, carty, (0, 0, 0))

        self.surf = pygame.transform.flip(self.surf, False, True)
        self.screen.blit(self.surf, (0, 0))
        if self.render_mode == "human":
            pygame.event.pump()
            self.clock.tick(self.metadata["render_fps"])
            pygame.display.flip()

        elif self.render_mode == "rgb_array":
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.screen)), axes=(1, 0, 2)
            )

    def close(self):
        if self.screen is not None:
            import pygame

            pygame.display.quit()
            pygame.quit()
            self.isopen = False