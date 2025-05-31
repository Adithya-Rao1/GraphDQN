import argparse

hp = argparse.Namespace()
hp.base_mols = {'C', 'O'}
hp.num_episodes = 100
hp.steps_per_episode = 20
hp.num_iterations = 1000
hp.num_epochs = 10
hp.rollout_length = 2000
hp.batch_size = 32
hp.gamma = 0.99
hp.lambda_ = 0.95
hp.epsilon = 0.2
hp.c1 = 0.5
hp.c2 = 0.01