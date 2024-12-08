import numpy as np
from numpy import cos, sin
from gymnasium.envs.classic_control.acrobot import AcrobotEnv

class AcrobotEnvWithSetState(AcrobotEnv):
    """
    state:  [
        theta1,
        theta2,
        theta1_dot,
        theta2_dot
    ]
    obs:    [
        cos(theta1),
        sin(theta1),
        cos(theta2),
        sin(theta2),
        theta1_dot,
        theta2_dot
    ]
    """
    name: str = "acrobot"

    def set_state(self, s: np.ndarray) -> None:
        """s is [theta1, theta2, theta1_dot, theta2_dot]."""
        self.state = s

        if self.render_mode == "human":
            self.render()

    def set_seed_states(self, seed_states: list[np.array]):
        self.seed_states = seed_states

    def remove_seed_states(self):
        self.seed_states = None

    def reset(self, *args, **kwargs):
        if hasattr(self, 'seed_states') and self.seed_states is not None:
            n = len(self.seed_states)
            chosen_idx = np.random.choice(n, size=1)[0]
            chosen_state = self.seed_states[chosen_idx]
            # print(f"Setting seed state {chosen_state}")
            self.set_state(chosen_state)
            return self._get_ob(), {}
        else:
            return super().reset(*args, **kwargs)
        
    def step(self, a):
        obs, reward, terminated, truncated, info = super().step(a)
        height = -cos(self.state[0]) - cos(self.state[1] + self.state[0])
        
        # 1.
        # reward = height
        
        # 2.
        # if height > 1:
        #     reward = 1
        # elif height > 0:
        #     reward = 0.
        # else:
        #     reward = -1

        reward += height * 0.01
        # if height > 1:
        #     reward = 0
        # else:
        #     reward = -1

        return obs, reward, terminated, truncated, info

    def get_observation(self, s: np.ndarray) -> np.ndarray:
        return np.array([cos(s[0]), sin(s[0]), cos(s[1]), sin(s[1]), s[2], s[3]], dtype=np.float32)