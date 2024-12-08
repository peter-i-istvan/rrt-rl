import os
import torch
import numpy as np
import pandas as pd
import scipy.special
from tqdm import tqdm
from scipy import spatial
import matplotlib.pyplot as plt
from dataclasses import dataclass

from dqn.model import Model
from env import CustomEnv


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return 1 - spatial.distance.cosine(a, b)


@dataclass(frozen=True)
class Transition:
    state: np.ndarray
    action: int
    next_state: np.ndarray
    reward: float
    done: bool


# noinspection PyAttributeOutsideInit
class RRTMemory:
    """Memory that fills itself based on a reference model passed to fill().
    During point randomization, it respects the upper and lower bounds provided in the constructor."""

    PLOTS_DIR = "TMP"

    def __init__(
        self,
        memory_size: int,
        batch_size: int,
        state_lower_bound: np.ndarray,
        state_upper_bound: np.ndarray,
        seed: int,
        save_folder: str,
        debug=True
    ) -> None:
        # sanity check
        assert memory_size > 10, "Memory size too low"
        # reproducibility
        np.random.seed(seed)
        # public state
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.state_lower_bound = state_lower_bound
        self.state_upper_bound = state_upper_bound
        self.debug = debug
        self.save_folder = save_folder
        # inner data container: storing buffer and tree as DF
        self._data = pd.DataFrame(
            columns=['state', 'action', 'next_state', 'reward', 'done']
        )
        self._tree = pd.DataFrame(
            columns=['hidden_state', 'state', 'q', 'parent', 'prev_action', 'avail_children', 'step']
        )
        # file operations
        os.makedirs(self.save_folder, exist_ok=True)
        if self.debug:
            os.makedirs(self.PLOTS_DIR, exist_ok=True)

    def _add(self, t: Transition) -> None:
        """Add transition tuple to inner container."""
        new_row = {
            'state': [t.state],
            'action': [t.action],
            'next_state': [t.next_state],
            'reward': [t.reward],
            'done': [t.done]
        }
        self._data = pd.concat([self._data, pd.DataFrame(new_row)], ignore_index=True)

    def _add_tree(
        self,
        hidden_state: np.ndarray,
        state: np.ndarray,
        q: np.ndarray,
        parent: int,
        prev_action: int,
        avail_children: int,
        done: bool,
        step: int
    ) -> None:
        """Add state to tree representation."""
        new_row = {
            'hidden_state': [hidden_state],
            'state': [state],
            'q': [q],
            'parent': [parent],
            'prev_action': [prev_action],
            'avail_children': [avail_children],
            'done': [done],
            'step': [step]
        }
        self._tree = pd.concat([self._tree, pd.DataFrame(new_row)], ignore_index=True)
        if parent >= 0:  # decrease # of available children for parent
            self._tree.loc[parent, 'avail_children'] -= 1

    def get_feature(self, state_a: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            state_t = torch.from_numpy(state_a.astype(np.float32))
            q_out = self.model(state_t)
            # TODO: proper comparison with softmax on/off
            return torch.softmax(q_out, dim=0).cpu().numpy()
            # return q_out.cpu().numpy()

    def get_random_state(self) -> np.ndarray:
        # TODO: try randomization in feature space?
        return np.random.uniform(self.state_lower_bound, self.state_upper_bound, (self.state_dim,))

    def get_closest_valid_node_index(self, random_q: np.ndarray) -> int:
        # valid = has at least one non-explored child and is not terminal => can be further expanded w/o redundancy
        valid_mask = (self._tree["avail_children"] > 0) & (~self._tree["done"].astype(bool))
        valid_states = self._tree.loc[valid_mask, "state"]
        valid_qs = self._tree.loc[valid_mask, "q"]

        if len(valid_states) == 0:
            self._tree.to_csv("tree.error.csv")
            raise ValueError("No more valid states found")

        # 1: cosine similarity metric
        cosine_sims = valid_qs.apply(lambda q: cosine_similarity(random_q, q))
        closest_idx = np.argmax(cosine_sims)  # 0-based index among valid states
        # 2: Euclidean distance metric - pass 'random_s' for this instead of 'random_q'
        # distances = valid_states.apply(lambda s: spatial.distance.euclidean(random_s, s))
        # closest_idx = np.argmin(distances)

        return valid_states.index[closest_idx]  # pos-based index in original DF

    # NODE EXPANSION STRATEGY A:
    #   ~ always moves towards 'better' (action with highest Q)
    def explore_new_tree_node_A(self, closest_node_index: int, stochastic: bool) -> int | None:
        # set start state for environment
        closest_tree_node = self._tree.iloc[closest_node_index]
        self.env.set_state(closest_tree_node["hidden_state"], step=closest_tree_node["step"])

        # filter out all actions that have been explored already
        action_space = np.arange(self.env.action_space.n)
        explored_actions = self._tree.loc[
            self._tree["parent"] == closest_node_index,
            "prev_action"
        ]
        mask = ~np.isin(
            action_space,  # all actions
            explored_actions
        )  # true for the elements of the action space that have not yet been explored

        # choose next action
        q = closest_tree_node["q"]
        if stochastic:
            p = scipy.special.softmax(q[mask])  # ~ probability distribution over available actions
            action = np.random.choice(action_space[mask], size=1, p=p)[0]
        else:
            available_actions = action_space[mask]
            action = available_actions[np.argmax(q[mask])]

        # step and assert terminal-ness
        next_state, reward, done, step = self.env.step(action, one_hot=False, return_step=True)

        if next_state is None:
            return None

        # add transition to train buffer
        closest_transition = Transition(closest_tree_node["state"], int(action), next_state, reward, done)
        self._add(closest_transition)

        # get hidden state
        next_hidden_state = self.env.state

        # add next state to tree
        self._add_tree(
            hidden_state=next_hidden_state,
            state=next_state,
            q=self.get_feature(next_state),
            parent=closest_node_index,
            prev_action=int(action),
            avail_children=self.env.action_space.n,
            done=done,
            step=step
        )

        return len(self._tree) - 1

    # TODO: rewrite
    # NODE EXPANSION STRATEGY B:
    #   ~ always moves 'closer' to target (cosine similarity to target_q in Q space)
    def explore_new_tree_node_B(
            self, closest_tree_state: np.ndarray, target_q: np.ndarray, stochastic: bool
    ) -> np.ndarray | None:
        """
        The next node is determined by inspecting all possible next states for all possible actions,
        based on these next states' closeness (cosine similarity) to 'target_q'.
        If 'stochastic' is True, we chose the action by random sampling from a distribution softmax(cosine_sims),
        otherwise we choose argmax(cosine_sims).
        In both cases, the resulting transition gets added to the train buffer in case of success.
        In case of failure (the chosen action resulted in NaN next state, or next state is duplicated),
        'None' is returned and no transition is added to the train buffer.
        """
        # Bálint: Here the environment should be set to the state defined by the exploration point
        # Then a method is required that defines how to choose an action from this exploration point
        # Then the environment's step() function should be called with the chosen action
        # Then the agent should predict the Q vector of S' (S' is the new node's state)
        # Then the new tree node shouldbe created from the gathered information S,A,R,S',D,Q,ES
        # Then the function must return the tree node in the data structure of choice

        transitions = []
        action_space = np.arange(self.env.action_space.n) # DiscreteActionSpace has attribute 'n'

        for action in action_space:
            self.env.set_state(closest_tree_state)
            next_state, reward, done = self.env.step(action, one_hot=False)
            # store transition for later
            transitions.append(
                Transition(
                    state=closest_tree_state,
                    action=action,
                    next_state=next_state,
                    reward=reward,
                    done=done
                )
            )

        # get Q scores - compare the "rolled out" == "next_state"s to the closest node
        # except terminal with no next_state
        transition_q_scores = [self.get_feature(t.next_state) for t in transitions if t.next_state is not None]

        if len(transition_q_scores) == 0:
            return None

        cosine_sims = [cosine_similarity(target_q, q) for q in transition_q_scores]

        if stochastic:
            p = scipy.special.softmax(cosine_sims)
            idx = np.random.choice(action_space, size=1, p=p)[0]
        else:
            idx = np.argmax(cosine_sims)

        closest_transition = transitions[idx]

        # TODO: add 'check_duplicate' flag to control whether we want to filter duplicates
        # TODO: more advanced duplicate checking - check nodes and its children - sample only among non-selected
        if self.is_repeated(closest_transition.next_state):
            return None

        self._add(closest_transition)  # adds to the buffer

        return closest_transition.next_state

    def is_repeated(self, state: np.ndarray) -> bool:
        # maybe change atol and rtol?
        return any(np.allclose(state, tree_state) for tree_state in self.tree_states)

    def fill(self, model: Model, env: CustomEnv, seed_states: list[np.ndarray]) -> None:
        """Builds an RRT memory from the seed_state, using the feature space (output) of the reference model."""
        # set up attributes:
        self.model = model
        self.env = env
        self.state_dim = len(env.state) # TODO: handle this properly
        # set up initial state
        for s in seed_states:
            obs = self.env.get_observation(s)
            self._add_tree(
                hidden_state=s,
                state=obs,
                q=self.get_feature(obs),
                parent=-1,
                prev_action=-1,
                avail_children=self.env.action_space.n,
                done=False,
                step=0
            )
        # set up other tracked values
        self.randomized_states = []
        self.randomized_q_values = []

        try:
            with tqdm(total=self.memory_size, desc='Filling RRT memory') as pbar:
                while len(self._data) < self.memory_size:
                    # random expansion
                    random_s = self.get_random_state()
                    random_s = env.get_observation(random_s)
                    random_q = self.get_feature(random_s)
                    closest_node_index = self.get_closest_valid_node_index(random_q)

                    # iterative expansion
                    prev_node_index = closest_node_index
                    for _ in range(5):
                        # A:
                        next_node_index = self.explore_new_tree_node_A(prev_node_index, stochastic=True)
                        # B:
                        # next_state = self.explore_new_tree_node_B(prev_state, target_q=q_random, stochastic=True)

                        if next_node_index is None:
                            continue

                        pbar.update(1)
                        # update the other lists too
                        self.randomized_states.append(random_s)
                        self.randomized_q_values.append(random_q)
                        # SLOW
                        # self.visualize(show_random=False, file_name_suffix=str(pbar.n), accentuate_last=True)
                        # move on to the next iteration, except if we reached a terminal node
                        if self._tree.iloc[next_node_index].done:
                            break

                        prev_node_index = next_node_index

        except KeyboardInterrupt as e:
            pass
        except ValueError:
            pass
        finally:
            self.save()

        # visualization
        if self.debug:
            print(f"{len(self._tree)=}")
            print(f"{len(self.randomized_states)=}")
            print(f"{len(self.randomized_q_values)=}")
            print(f"{len(self._data)=}")
            # self.visualize()

    def save(self):
        # states (observations) as stacked vectors
        np.save(
            os.path.join(self.save_folder, "states.rrt.npy"),
            np.stack(self._data.state.tolist())
        )
        self._data.to_csv(os.path.join(self.save_folder, "samples.rrt.csv"), index=False)
        self._tree.to_csv(os.path.join(self.save_folder, "tree.csv"))

    def visualize(self, show_random: bool = True, accentuate_last: bool = False, file_name_suffix: str = "") -> None:
        # visualize in Q space
        if show_random:
            q0, q1 = zip(*self.randomized_q_values)
            plt.scatter(q0, q1, label="random Q values")

        tree_q_values = self._tree["q"].values
        q0, q1 = zip(*tree_q_values)
        plt.scatter(q0, q1, label="tree Q values")

        if accentuate_last:
            q0, q1 = tree_q_values[-1]
            plt.scatter([q0], [q1], label="last tree Q value", color="red")

        plt.xlabel("q0")
        plt.ylabel("q1")
        plt.title("Points in memory (Q-space)")
        plt.legend()
        plt.savefig(os.path.join(self.PLOTS_DIR, f"RRT-Q-space{file_name_suffix}.png"))
        plt.clf()

        # visualize in state space
        fig, axs = plt.subplots(nrows=1, ncols=2)

        if show_random:
            x, x_dot, theta, theta_dot = zip(*self.randomized_states)
            axs.flat[0].scatter(x, x_dot, label="random states (pos+velocity)")
            axs.flat[1].scatter(theta, theta_dot, label="random states (angle+velocity)")

        tree_states = self._tree["state"].values
        x, x_dot, theta, theta_dot = zip(*tree_states)
        axs.flat[0].scatter(x, x_dot, label="tree states (pos+velocity)")
        axs.flat[1].scatter(theta, theta_dot, label="tree states (angle+velocity)")

        if accentuate_last:
            x, x_dot, theta, theta_dot = tree_states[-1]
            axs.flat[0].scatter([x], [x_dot], label="last tree state", color="red")
            axs.flat[1].scatter([theta], [theta_dot], label="last tree state", color="red")

        axs.flat[0].set_xlabel("position")
        axs.flat[0].set_ylabel("velocity")
        axs.flat[1].set_xlabel("angle")
        axs.flat[1].set_ylabel("angular velocity")

        fig.suptitle("Points in memory (State space)")
        fig.legend()
        fig.savefig(os.path.join(self.PLOTS_DIR, f"RRT-state-space{file_name_suffix}.png"))
        fig.clf()

        plt.close()

    def load_samples(self):
        # NOTE: this does not work
        self._data = pd.read_csv("samples.rrt.csv")

    def sample(self):
        batch = self._data.sample(self.batch_size)  # TODO: seeding!

        def transform(k: str, dtype=np.float32) -> torch.Tensor:
            return torch.from_numpy(np.array(batch[k].tolist()).astype(dtype))

        state = transform("state")
        action = transform("action", dtype=np.int64)
        next_state = transform("next_state")
        reward = transform("reward")
        done = transform("done")

        return state, action, reward, next_state, done
