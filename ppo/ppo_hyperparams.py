import argparse

hp = argparse.Namespace()
hp.base_mols = {'C', 'O'}
hp.steps_per_episode = 100
hp.num_episodes = 500


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--steps_per_episode', type=int, default=hp.steps_per_episode)
    parser.add_argument('--num_episodes', type=int, default=hp.num_episodes)
    args = parser.parse_args()
    return args