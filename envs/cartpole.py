import numpy as np
from gymnasium.envs.classic_control import CartPoleEnv


class CartPoleEnvWithSetState(CartPoleEnv):
    name: str = "cartpole"
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.seed_states = None

    def set_seed_states(self, seed_states: list[np.array]):
        self.seed_states = seed_states

    def remove_seed_states(self):
        self.seed_states = None

    def reset(self, *args, **kwargs):
        if self.seed_states is not None:
            n = len(self.seed_states)
            chosen_idx = np.random.choice(n, size=1)[0]
            chosen_state = self.seed_states[chosen_idx]
            # print(f"Setting seed state {chosen_state}")
            self.set_state(chosen_state)
            return chosen_state, {}
        else:
            return super().reset(*args, **kwargs)

    def set_state(self, state: np.ndarray) -> None:
        """Similar to reset(), but with a specific state instead of random."""
        self.state = state
        # Boilerplate copied from CartPoleEnv.reset implementation
        # (https://github.com/openai/gym/blob/master/gym/envs/classic_control/cartpole.py)
        self.steps_beyond_terminated = None

        if self.render_mode == "human":
            self.render()

    def get_observation(self, s: np.ndarray) -> np.ndarray:
        return s
