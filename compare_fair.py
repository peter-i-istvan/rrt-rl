import copy
import matplotlib.pyplot as plt
from argparse import ArgumentParser

from utils import *
from env import CustomEnv
from dqn.model import Model
from memory.rrt import RRTMemory
from memory.stochastic import Buffer
from envs import CartPoleEnvWithSetState, AcrobotEnvWithSetState, MountainCarEnvWithSetState
from dqn.dqn_agent import DeepQNetworkAgent as Agent


def get_args():
    parser = ArgumentParser(description='Configuration file for training')
    parser.add_argument('--config', '-c', type=str, default='cartpole-rrt.ini', help='Configuration file')
    return parser.parse_args()


def setup():
    args = get_args()
    config = read(args.config)
    seed = config.enviroment.seed
    set_seeds(seed)
    return config


def get_seeded_env(config):
    # get env by name
    if 'cartpole' in config.enviroment.name.lower():
        inner_env = CartPoleEnvWithSetState()
    elif 'acrobot' in config.enviroment.name.lower():
        inner_env = AcrobotEnvWithSetState()
    elif 'mountain_car' in config.enviroment.name.lower():
        inner_env = MountainCarEnvWithSetState()
    else:
        raise ValueError(f"{config.enviroment.name} env not supported")

    # set seed states
    seed_states = []
    for i in range(config.enviroment.n_seed_states):
        inner_env.reset(seed=i)
        s = inner_env.state
        seed_states.append(s)

    print(f"Setting {config.enviroment.n_seed_states} seed states")
    inner_env.set_seed_states(seed_states)

    return inner_env, seed_states


def train_simple(config, env):
    model = Model(
        in_features=env.in_features,
        hidden_features=config.model.hidden_features,
        out_features=env.out_features,
        hidden_activation=config.model.hidden_activation,
        out_activation=config.model.out_activation,
        lr=config.model.lr,
    )
    memory = Buffer(
        env=env,
        memory_size=config.memory.size,
        batch_size=config.memory.batch_size,
        save_folder=os.path.join("output", env.name)
    )
    agent = Agent(
        model=model,
        target_model=copy.deepcopy(model),
        epsilon=1,
        discount_factor=config.agent.discount_factor,
        epsilon_decay=config.agent.epsilon_decay,
        target_update_frequency=config.agent.target_update_frequency,
        save_folder=os.path.join("output", env.name)
    )
    memory.fill(agent)

    # memory is filled, seeding can be turned off temporarily
    # to allow validation to work with more than the seed states
    seed_states = env.seed_states
    env.remove_seed_states()
    agent.train_with_interaction(
        env=env,
        memory=memory,
        start=0,
        end=config.mcts.train_episodes,
        verbose=False
    )

    print(agent.validate(env))
    # revert to seeded state
    env.set_seed_states(seed_states)

    return agent.val_rewards


def train_rrt(config, env, seed_states, master_model):
    rrt_memory = RRTMemory(
        memory_size=config.rrt.size,
        batch_size=config.memory.batch_size,
        state_lower_bound=np.array(config.rrt.state_lower_bound),
        state_upper_bound=np.array(config.rrt.state_upper_bound),
        seed=config.enviroment.seed,
        debug=True,
        save_folder=os.path.join("output", env.name)
    )
    rrt_memory.fill(master_model, env, seed_states)
    model = Model(
        in_features=env.in_features,
        hidden_features=config.model.hidden_features,
        out_features=env.out_features,
        hidden_activation=config.model.hidden_activation,
        out_activation=config.model.out_activation,
        lr=config.model.lr,
    )
    agent = Agent(
        model=model,
        target_model=copy.deepcopy(model),
        epsilon=1,
        discount_factor=config.agent.discount_factor,
        epsilon_decay=config.agent.epsilon_decay,
        target_update_frequency=config.agent.target_update_frequency,
        save_folder=os.path.join("output", env.name)
    )

    # memory is filled, seeding can be turned off temporarily
    # to allow validation to work with more than the seed states
    seed_states = env.seed_states
    env.remove_seed_states()

    agent.train_with_interaction(
        env=env,
        memory=rrt_memory,
        start=0,
        end=config.mcts.train_episodes,
        verbose=False,
        model_kind='rrt'
    )

    # revert to seeded state
    env.set_seed_states(seed_states)

    return agent.val_rewards


def main():
    config = setup()
    inner_env, seed_states = get_seeded_env(config)
    env = CustomEnv(inner_env, max_steps=config.enviroment.max_steps)
    
    simple_val_rewards = train_simple(config, env)
    # read master model (best performing checkpoint from prev. train)
    master_model = torch.load(
        os.path.join("output", env.name, "model.dqn.pth")
    )
    master_model.model.eval()

    rrt_val_rewards = train_rrt(config, env, seed_states, master_model)

    plt.plot(simple_val_rewards, label=f"baseline [{config.memory.size} samples]")
    plt.plot(rrt_val_rewards, label=f"RRT [{config.rrt.size} samples]")
    plt.title(f"Validation rewards ({len(seed_states)} seed states)")
    plt.legend()
    plt.savefig(os.path.join("output", env.name, "val_rewards.png"))


if __name__ == '__main__':
    main()
