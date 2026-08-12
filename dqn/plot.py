import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from rdkit import Chem
from rdkit.Chem import QED 
from dqn.utils import penalized_logp

from ADMET.model import ADMETModel
from binding_module.binding_affinity.plapt import Plapt, run_predictions
from binding_module.selectivity.compare import compare_affinities
from synthetic_accessibility.sa_score import SyntheticAccessibility
from experiments.data.targets import TARGETS, DEFAULT_TARGET


def plot_molecule_distributions(molecules, filename="molecule_distributions.png", target_name=DEFAULT_TARGET):
    target_seq = TARGETS[target_name]

    logp_scores = [penalized_logp(Chem.MolFromSmiles(mol)) for mol in molecules]
    qed_scores = [QED.qed(Chem.MolFromSmiles(mol)) for mol in molecules]
    binding_affinities = run_predictions(Plapt(), target_seq, molecules)  # Assuming this returns a list
    synthetic_accessibility = [SyntheticAccessibility().calculateScore(Chem.MolFromSmiles(mol)) for mol in molecules]

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    sns.kdeplot(logp_scores, ax=axes[0, 0], fill=True, bw_adjust=0.5)
    axes[0, 0].set_title("LogP Score Distribution")

    sns.kdeplot(qed_scores, ax=axes[0, 1], fill=True, bw_adjust=0.5)
    axes[0, 1].set_title("QED Score Distribution")

    sns.kdeplot(binding_affinities, ax=axes[1, 0], fill=True, bw_adjust=0.5)
    axes[1, 0].set_title("Binding Affinity Distribution")

    sns.kdeplot(synthetic_accessibility, ax=axes[1, 1], fill=True, bw_adjust=0.5)
    axes[1, 1].set_title("Synthetic Accessibility Score Distribution")

    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()

    print(f"Graph saved as {filename}")

if __name__ == "__main__":
    import argparse

    default_smiles = [
            'N#CC1=C(C(F)(F)F)C([N+]([O-])=O)=C(C2=CC=CC([N+]([O-])=O)=C2)NC1=O',
            'O=C(NC1=CC=C(NC(NC2=CC=CC3=C2C=CN3)=O)C=C1)NC4=C(C=CN5)C5=CC=C4',
            'O=C1N(C)C(CNCC)=NC2=C1C(Cl)=CC(Cl)=C2O.Br',
            'O=C(C(C=C1C2=O)=CC(O)=C1C(C3=C2C=CC=C3O)=O)NC4=CC=C(Cl)C=C4O',
            'OC1=CC=C(CC2=CC=C(C(CC3=CC=C(C(CC4=CC=C(C=C4)O)=C3)O)=C2)O)C=C1'
            ]

    parser = argparse.ArgumentParser()
    parser.add_argument('--smiles-path', type=str, default=None,
                         help='Path to a newline-delimited file of SMILES. Defaults to a small built-in example set.')
    parser.add_argument('--target-name', type=str, default=DEFAULT_TARGET, choices=list(TARGETS))
    parser.add_argument('--filename', type=str, default='molecule_distributions.png')
    args = parser.parse_args()

    if args.smiles_path:
        with open(args.smiles_path, 'r') as f:
            smiles_list = f.read().splitlines()
    else:
        smiles_list = default_smiles

    plot_molecule_distributions(smiles_list, filename=args.filename, target_name=args.target_name)
