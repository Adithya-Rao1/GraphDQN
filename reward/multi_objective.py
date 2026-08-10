from typing import Optional

from rdkit import Chem

from ADMET.model import ADMETModel
from binding_module.binding_affinity.plapt import Plapt, run_predictions
from binding_module.selectivity.compare import compare_affinities
from synthetic_accessibility.sa_score import SyntheticAccessibility

ADMET_PROPERTIES = [
    "QED",
    "Lipinski",
    "Bioavailability_Ma",
    "BBB_Martins",
    "DILI",
    "Clearance_Hepatocyte_AZ",
    "Clearance_Microsome_AZ",
    "Half_Life_Obach",
    "hERG",
    "ClinTox",
    "LD50_Zhu",
]

ADMET_OPTIM_DIRECTIONS = [1, 1, 1, 1, -1, 1, 1, 1, -1, -1, -1]


def compute_admet_reward(admet_preds: dict) -> float:
    return sum(
        admet_preds[prop] if direction == 1 else (1 / admet_preds[prop])
        for prop, direction in zip(ADMET_PROPERTIES, ADMET_OPTIM_DIRECTIONS)
    )


def compute_reward(
    smiles: str,
    target_seq: str,
    device,
    off_target_seq: Optional[str] = None,
    admet_weight: float = 0.3,
    binding_weight: float = 0.5,
    synthetic_weight: float = 0.2,
    selectivity_weight: float = 0.3,
    admet_model: Optional[ADMETModel] = None,
    binding_model: Optional[Plapt] = None,
    sa_model: Optional[SyntheticAccessibility] = None,
) -> dict:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {"reward": 0.0, "admet": 0.0, "binding_uM": None, "sa_score": None, "selectivity": None}

    admet_model = admet_model if admet_model is not None else ADMETModel(device)
    binding_model = binding_model if binding_model is not None else Plapt(device=str(device))
    sa_model = sa_model if sa_model is not None else SyntheticAccessibility()

    admet_preds = admet_model.predict(smiles)
    admet_reward = compute_admet_reward(admet_preds)

    binding_uM = run_predictions(binding_model, target_seq, [smiles])[0]
    sa_score = sa_model.calculateScore(mol)

    if off_target_seq:
        off_target_uM = run_predictions(binding_model, off_target_seq, [smiles])[0]
        selectivity = compare_affinities(binding_uM, off_target_uM)

        scale = selectivity_weight / 3
        reward = (
            (admet_weight - scale) * admet_reward
            + (binding_weight - scale) * (1 / binding_uM)
            + (synthetic_weight - scale) * (1 / sa_score)
            + selectivity_weight * selectivity
        )
    else:
        selectivity = None
        reward = (
            admet_weight * admet_reward
            + binding_weight * (1 / binding_uM)
            + synthetic_weight * (1 / sa_score)
        )

    return {
        "reward": reward,
        "admet": admet_reward,
        "binding_uM": binding_uM,
        "sa_score": sa_score,
        "selectivity": selectivity,
    }
