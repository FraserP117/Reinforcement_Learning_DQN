import argparse, os
import numpy as np
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from replay_memory import ReplayBuffer
from preprocessing import make_environment
from utils import plot_learning_curve
import agents as Agents
from dqn import DQNetwork
from gym import wrappers


def main():

    parser = argparse.ArgumentParser(description = "OG DQN Algrithm")

    parser.add_argument('-n_games', type = int, default = 1, help = 'Number of games to play')
    parser.add_argument('-lr', type = float, default = 0.0001, help = 'Learning Rate for Optimizer')
    parser.add_argument('-eps_min', type = float, default = 0.1, help = 'Minimim value for epsilon')
    parser.add_argument('-gamma', type = float, default = 0.99, help = 'Discount factor')
    parser.add_argument('-eps_dec', type = float, default = 1e-5, help = 'Linear factor for decreasing epsilon')
    parser.add_argument('-eps', type = float, default = 1.0, help = 'Starting value for epsilon')
    parser.add_argument('-max_mem', type = int, default = 30000, help = 'Max size for memory buffer')
    parser.add_argument('-skip', type = int, default = 4, help = 'Number frames to stack into environment observations')
    parser.add_argument('-bs', type = int, default = 32, help = 'Batch size for replay memory sampling')
    parser.add_argument('-replace', type = int, default = 1000, help = 'Interval for replacing the target network')
    parser.add_argument('-env', type = str, default = 'PongNoFrameskip-v4', help = 'Atari environment. \nPongNoFrameskip-v4\n \
                                                                                    BreakoutNoFrameskip-v4\n \
                                                                                    EnduroNoFrameskip-v4\n \
                                                                                    AtlantisNoFrameskip-v4')
    parser.add_argument('-gpu', type = str, default = '0', help = 'GPU: 0 or 1')
    parser.add_argument('-load_checkpoint', type = bool, default = False, help = 'Load model checkpoint')
    parser.add_argument('-path', type = str, default = 'tmp/dqn', help = 'Path for model Saving/Loading')
    parser.add_argument('-algo', type = str, default = 'DQNAgent', help = 'DQNAgent/DoubleDQNAgent/DuelingDQNAgent/DuelingDoubleDQNAgent')

    args = parser.parse_args()

    print("Hyperparameters: ")
    for arg in vars(args):
        print(f"{arg}: {getattr(args, arg)}")
    print("---------------------------------------------")

    os.environ['CUDA DEVICE ORDER'] = 'PCI BUS ID'
    os.environ['CUDA VISIBLE DEVICES'] = args.gpu

    print("Creating the environment...")
    env = make_environment(args.env)
    best_score = -np.inf

    print(f"Instantiating the {args.algo}...")
    atari_agent = getattr(Agents, args.algo)
    agent = atari_agent(
        gamma = args.gamma,
        epsilon = args.eps,
        lr = args.lr,
        input_dims = env.observation_space.shape,
        n_actions = env.action_space.n,
        mem_size = args.max_mem,
        eps_min = args.eps_min,
        eps_dec = args.eps_dec,
        batch_size = args.bs,
        replace_interval = args.replace,
        chkpt_dir = args.path,
        algo = args.algo,
        env_name = args.env
    )

    if args.load_checkpoint:
        agent.load_models()

    env = wrappers.Monitor(env, 'tmp/video', video_callable = lambda episode_id: True, force = True)

    fname = f"algo_{args.algo}_env_{args.env}_lr_{str(args.lr)}_n_games_{str(args.n_games)}"
    figure_file = 'plots/' + fname + '.png'

    n_steps = 0
    scores, eps_history, steps_arr = [], [], []

    print(f"Beginning exec of {args.n_games} games...")

    for i in range(args.n_games):
        done = False
        score = 0
        observation = env.reset()

        while not done:
            action = agent.choose_action(observation)
            new_observation, reward, done, info = env.step(action)
            score += reward

            # Visualise the agent's performance in the environment:
            if args.load_checkpoint:
                env.render()

            # if learning
            if not args.load_checkpoint:
                agent.store_transition(observation, action, reward, new_observation, int(done))
                agent.learn(observation, action, reward, new_observation)

            observation = new_observation
            n_steps += 1

        scores.append(score)
        steps_arr.append(n_steps)
        eps_history.append(agent.epsilon)

        avg_score_100 = np.mean(scores[-100:])
        print(f"episode: {i}, score: {score}, AVG_score: {avg_score_100:.1f}, epsilon: {agent.epsilon:.2f}, total_steps: {n_steps}")

        if avg_score_100 > best_score:
            if not args.load_checkpoint:
                agent.save_models()
            best_score = avg_score_100

    plot_learning_curve(steps_arr, scores, eps_history, figure_file)

if __name__ == '__main__':
    main()
