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


def plot_molecule_distributions(molecules, filename="molecule_distributions.png"):
    # Compute scores
    target_seq = 'MDVFMKGLSKAKEGVVAAAEKTKQGVAEAAGKTKEGVLYVGSKTKEGVVHGVATVAEKTKEQVTNVGGAVVTGVTAVAQKTVEGAGSIAAATGFVKKDQLGKNEEGAPQEGILEDMPVDPDNEAYEMPSEEGYQDYEPEA'

    logp_scores = [penalized_logp(Chem.MolFromSmiles(mol)) for mol in molecules]
    qed_scores = [QED.qed(Chem.MolFromSmiles(mol)) for mol in molecules]
    binding_affinities = run_predictions(Plapt(), target_seq, molecules)  # Assuming this returns a list
    synthetic_accessibility = [SyntheticAccessibility().calculateScore(Chem.MolFromSmiles(mol)) for mol in molecules]

    # Create subplots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # KDE Plots
    sns.kdeplot(logp_scores, ax=axes[0, 0], fill=True, bw_adjust=0.5)
    axes[0, 0].set_title("LogP Score Distribution")

    sns.kdeplot(qed_scores, ax=axes[0, 1], fill=True, bw_adjust=0.5)
    axes[0, 1].set_title("QED Score Distribution")

    sns.kdeplot(binding_affinities, ax=axes[1, 0], fill=True, bw_adjust=0.5)
    axes[1, 0].set_title("Binding Affinity Distribution")

    sns.kdeplot(synthetic_accessibility, ax=axes[1, 1], fill=True, bw_adjust=0.5)
    axes[1, 1].set_title("Synthetic Accessibility Score Distribution")

    # Adjust layout and save the figure
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()

    print(f"Graph saved as {filename}")

"""smiles_list = [
        'N#CC1=C(C(F)(F)F)C([N+]([O-])=O)=C(C2=CC=CC([N+]([O-])=O)=C2)NC1=O',
        'O=C(NC1=CC=C(NC(NC2=CC=CC3=C2C=CN3)=O)C=C1)NC4=C(C=CN5)C5=CC=C4',
        'O=C1N(C)C(CNCC)=NC2=C1C(Cl)=CC(Cl)=C2O.Br',
        'O=C(C(C=C1C2=O)=CC(O)=C1C(C3=C2C=CC=C3O)=O)NC4=CC=C(Cl)C=C4O',
        'OC1=CC=C(CC2=CC=C(C(CC3=CC=C(C(CC4=CC=C(C=C4)O)=C3)O)=C2)O)C=C1'
        ]"""

smiles_path = 'Metis_Data/C_FactorVAE_Data/all.txt'
with open(smiles_path, 'r') as f:
    smiles_list = f.read().splitlines()

plot_molecule_distributions(smiles_list)
