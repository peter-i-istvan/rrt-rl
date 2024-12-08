import numpy as np
from numpy import cos, sin
from gymnasium.envs.classic_control.mountain_car import MountainCarEnv


class MountainCarEnvWithSetState(MountainCarEnv):
    """state:  [position, velocity] (=obs)"""
    name: str = "mountain_car"

    def set_state(self, s: np.ndarray) -> None:
        """s is [position, velocity]."""
        self.state = s.astype(np.float32)

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
            return self.state.astype(np.float32), {}
        else:
            return super().reset(*args, **kwargs)

    def step(self, a):
        obs, reward, terminated, truncated, info = super().step(a)
        self.state = np.array(self.state)
        obs = np.array(obs, dtype=np.float32)
        # override reward here
        # 1.
        # reward += 10. * abs(self.state[1])
        # reward = 10. * abs(self.state[1])

        # 2. gmh + mv^2/2 = m(gh + v^2/2), m fix (:=1)
        h = float(self._height(self.state[0]))
        v = self.state[1]
        reward += 10*(self.gravity * h + v**2/2)

        # 3. large reward when reaching goal
        # reward -= abs(self.goal_position - self.state[0])
        return obs, reward, terminated, truncated, info

    def get_observation(self, s: np.ndarray) -> np.ndarray:
        return s