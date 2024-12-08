import os
import torch
import random
import numpy as np
from tqdm import tqdm
from time import time
import torch.nn.functional as F


class DeepQNetworkAgent():
    def __init__(self,model, target_model, epsilon, discount_factor, epsilon_decay, target_update_frequency, save_folder):
        super().__init__()
        self.model = model
        self.target_model = target_model
        self.target_model.eval()
        
        self.epsilon = epsilon
        self.discount_factor = discount_factor
        self.epsilon_decay = epsilon_decay
        self.target_update_frequency = target_update_frequency
        # folder ops
        self.save_folder = save_folder
        os.makedirs(self.save_folder, exist_ok=True)
        
    def train_predict(self, state: np.ndarray) -> int:
        if self.epsilon > random.random():
            p =  np.random.randint(0, self.model.out_features, 1)[0]
          
        else:
            p = self.inference_predict(state)

        return p

    def inference_predict(self, state: np.ndarray):
        self.model.eval()

        with torch.no_grad():
            actions = self.model(torch.reshape(torch.tensor(state, dtype=torch.float32), shape=(1, self.model.in_features)))
            p = int(torch.argmax(actions))

        self.model.train()

        return p

    def decay_epsilon(self):
        self.epsilon *= self.epsilon_decay

    def fit(self, memory):

        states, actions, rewards, next_states, terminals = memory.sample()
        states = states.to(torch.float32)
        predicted_q = self.model(states)
        predicted_q = torch.gather(predicted_q, 1, actions.view((-1, 1))).squeeze()

      
        with torch.no_grad():
            target_q = self.target_model(torch.squeeze(next_states))
            target_q = target_q.max(dim=-1)[0]
        target_value = rewards + self.discount_factor * target_q * (1 - terminals)

        loss = F.mse_loss(predicted_q, target_value)

        self.model.optimizer.zero_grad()
        loss.backward()

        self.model.optimizer.step()
         
    def train_no_interaction(self, env, memory, start, end, verbose=False):
        print(f"Training from episode {start} to episode {end}")

        for episode in tqdm(range(start, end)):
            self.fit(memory)

            if episode % self.target_update_frequency == 0:
                self.update_target_network()
        
    def train_with_interaction(self, env, memory, start, end, verbose=False, model_kind='dqn'):
        print(f"Training from episode {start} to episode {end}")
        episode_rewards = []
        self.val_rewards = []
        best_val_reward = None
        end_training = False

        for episode in (pbar := tqdm(range(start, end))):
            state = env.reset()
            rewards = 0

            steps = 0
            while True:
                action = self.train_predict(state)
                next_state, reward, done = env.step(action)

                # memory.add(state, action, reward, next_state, done)

                rewards += reward

                state = next_state

                if done:
                    # EARLY STOPPING
                    val_reward = rewards

                    if best_val_reward is None or val_reward > best_val_reward:
                        best_val_reward = val_reward
                        self.save_checkpoint(model_kind)

                    pbar.set_description(
                        f"R: {val_reward:.2f} Best: {best_val_reward:.2f} ε: {self.epsilon:.2f}"
                    )
                    episode_rewards.append(rewards)
                    self.val_rewards.append(val_reward)

                    self.fit(memory)
                    self.decay_epsilon()
                    
                    if (episode + 1) % self.target_update_frequency == 0:
                        self.update_target_network()

                    break

                steps += 1
            
            if end_training:
                break

        self.save_rewards(episode_rewards, model_kind)
        if verbose:
            print(f"Episode {episode+1}/{end}, rewards: {rewards}, epsilon: {self.epsilon}")

    def save_checkpoint(self, model_kind):
        torch.save(
            self.model,
            os.path.join(self.save_folder, f'model.{model_kind}.pth')
        )
    
    def save_rewards(self, episode_rewards, model_kind):
        np.save(
            os.path.join(self.save_folder, f'rewards.{model_kind}.npy'),
            episode_rewards
        )

    def validate(self,env):
        rewards = 0
        
        state = env.reset()

        while True:
            action = self.inference_predict(state)
            next_state, reward, done = env.step(action)

            # env.render()

            state = next_state

            if reward == -100:
                pass

            rewards += reward

            if done:
                break

        return rewards

    def update_target_network(self):
        self.target_model.load_state_dict(self.model.state_dict())
