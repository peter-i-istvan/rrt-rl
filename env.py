import numpy as np
import gymnasium as gym


class Env:
    def __init__(self, env_name, max_steps=None, render_mode=None, seed=42):
        self.seed = seed
        self.max_steps = max_steps
        self._start_steps = 0
        try:
            self.gym_env = gym.make(env_name, render_mode=render_mode)
        except gym.error.NameNotFound:
            raise ValueError(f"{env_name} environment is not implemented or"
                                f" not found in the official gym repository")
        
    @property
    def in_features(self):
        try:
            return self.gym_env.observation_space.shape[0]
        except:
            return self.gym_env.observation_space.n
    
    @property
    def out_features(self):
        try:
            return self.gym_env.action_space.shape[0]
        except:
            return self.gym_env.action_space.n
        
    def _one_hot(self, state):
        if isinstance(state, int):
            one_hot = np.zeros(self.in_features)
            one_hot[state] = 1

            return one_hot.astype(np.float32)
            # return np.float32(state)
        else:
            return state

    def reset(self):
        state, _ = self.gym_env.reset(seed=self.seed)

        self._start_steps = 0
        
        return self._one_hot(state)
        # return state

    def step(self, action, one_hot=True):
        next_state, reward, terminated, truncated, _ = self.gym_env.step(action)

        self._start_steps += 1

        if self.max_steps is not None and self._start_steps >= self.max_steps:
            truncated = True

        done = terminated or truncated

        if one_hot:
            return self._one_hot(next_state), reward, done
        else:
            return next_state, reward, done

    def render(self):
        self.gym_env.render()

    def close(self):
        self.gym_env.close()

    # def __del__(self):
    #     self.gym_env.close()


class CustomEnv(Env):
    # noinspection PyMissingConstructor
    def __init__(self, env, max_steps=None, render_mode=None, seed=42):
        self.seed = seed
        self.max_steps = max_steps
        self._start_steps = 0
        self.gym_env = env

    @property
    def state(self) -> np.ndarray:
        return self.gym_env.state

    @property
    def action_space(self):
        return self.gym_env.action_space

    @property
    def name(self):
        return self.gym_env.name

    def set_seed_states(self, seed_states: list[np.array]):
        self.gym_env.set_seed_states(seed_states)

    def remove_seed_states(self):
        self.gym_env.remove_seed_states()

    @property
    def seed_states(self) -> list[np.ndarray]:
        return self.gym_env.seed_states

    def set_state(self, state: np.ndarray, step: int) -> None:
        self._start_steps = step
        self.gym_env.set_state(state)

    def step(self, action, one_hot=True, return_step=False) -> tuple:
        """Additionally returns the current step no."""
        ret = super().step(action, one_hot)
        if return_step:
            return *ret, self._start_steps
        else:
            return ret
        
    def get_observation(self, s: np.ndarray) -> np.ndarray:
        return self.gym_env.get_observation(s)
