import random

from all_experiments.smiles_800 import start_mols as ALL_STARTING_MOLECULES


def sample_pilot_molecules(n=30, seed=0):
    rng = random.Random(seed)
    return rng.sample(ALL_STARTING_MOLECULES, min(n, len(ALL_STARTING_MOLECULES)))
