import argparse
import ppo_hyperparams as hp

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_iterations', type=int, default=hp.num_iterations, help='Number of training iterations')
    parser.add_argument('--num_epochs', type=int, default=hp.num_epochs, help='Number of training epochs')
    parser.add_argument('--rollout_length', type=int, default=hp.rollout_length, help='Number of steps per iteration')
    parser.add_argument('--batch_size', type=int, default=hp.batch_size, help='Training batch size')
    parser.add_argument('--gamma', type=float, default=hp.gamma, help='Discount factor')
    parser.add_argument('--lambda_', type=float, default=hp.lambda_, help='GAE hyperparameter')
    parser.add_argument('--epsilon', type=float, default=hp.epsilon, help='Clipping parameter')
    parser.add_argument('--c1', type=float, default=hp.c1, help='Value loss coefficient')
    parser.add_argument('--c2', type=float, default=hp.c2, help='Entropy loss coefficient')

    args = parser.parse_args()
    return args