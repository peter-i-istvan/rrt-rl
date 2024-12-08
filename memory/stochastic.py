import os
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm


class Buffer:
    def __init__(self, env, memory_size, batch_size, save_folder):
        self.env = env
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.save_folder = save_folder
        # inner state container
        self.data = pd.DataFrame(columns=['state', 'action', 'next_state', 'reward', 'done'])
        # folder ops
        os.makedirs(self.save_folder, exist_ok=True)


    def _add(self, state, action, reward, next_state,  done):
        new_row = {'state': state, 'action': action, 'next_state': next_state, 'reward': reward, 'done': done}

        self.data = pd.concat([self.data, pd.DataFrame([new_row])], ignore_index=True)

    def add(self, state, action, reward, next_state, done):
        self.data = self.data.iloc[1:]

        self._add(state, action, reward, next_state, done)

    def sample(self):
        # sample only the last memory_size elements (circular buffer)
        batch = self.data[-self.memory_size:].sample(self.batch_size)

        # Convert batch DataFrame to numpy array
        batch_np = batch.to_numpy()

        # Extract columns from the numpy array
        # state = torch.tensor(np.vstack(batch_np[:, 0]))
        # action = torch.tensor(np.vstack(batch_np[:, 1]))
        # next_state = torch.tensor(np.vstack(batch_np[:, 2]))
        # reward = torch.tensor(np.hstack(batch_np[:, 3]))
        # done = torch.LongTensor(np.hstack(batch_np[:, 4]))
        state = torch.tensor(batch_np[:, 0].tolist())
        action = torch.tensor(batch_np[:, 1].tolist())
        next_state = torch.tensor(batch_np[:, 2].tolist())
        reward = torch.tensor(batch_np[:, 3].tolist())
        done = torch.LongTensor(batch_np[:, 4].tolist())

        return state, action, reward, next_state, done

    def fill(self, agent):
        num_of_samples = 0

        with tqdm(total=self.memory_size, desc="Filling memory") as pbar:
            while num_of_samples < self.memory_size:
                state = self.env.reset()

                while True:
                    action = agent.train_predict(state)

                    next_state, reward, done = self.env.step(action)

                    self._add(state=state, action=action, reward=reward, next_state=next_state, done=done)

                    num_of_samples += 1
                    pbar.update(1)

                    state = next_state

                    if done or num_of_samples == self.memory_size:
                        break

        self.data = self.data.sample(frac=1, random_state=42).reset_index(drop=True)
        self.save()
        print(f'Filled memory with {num_of_samples} samples')

    def save(self):
        np.save(
            os.path.join(self.save_folder, "states.baseline.npy"),
            np.stack(self.data.state.tolist())
        )
        self.data.to_csv(os.path.join(self.save_folder, "samples.baseline.csv"))
        