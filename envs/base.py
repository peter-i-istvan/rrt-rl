import numpy as np
from abc import ABC, abstractmethod


class EnvWithSetState(ABC):
    @abstractmethod
    def set_state(self, state: np.ndarray) -> None:
        raise NotImplementedError()
