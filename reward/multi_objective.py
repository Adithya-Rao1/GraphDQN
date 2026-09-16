from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence

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

ADMET_SCALE = {
    "Lipinski": 2.0,
    "Clearance_Hepatocyte_AZ": 50.0,
    "Clearance_Microsome_AZ": 50.0,
    "Half_Life_Obach": 24.0,
    "LD50_Zhu": 2.5,
}


def _normalize_admet_property(prop: str, value: float, direction: int) -> float:
    if prop in ADMET_SCALE:
        value = value / (value + ADMET_SCALE[prop]) if value >= 0 else 0.0
    return value if direction == 1 else (1.0 - value)


def compute_admet_reward(
    admet_preds: dict,
    properties: Optional[Sequence[str]] = None,
    directions: Optional[Dict[str, int]] = None,
) -> float:
    properties = list(properties) if properties is not None else ADMET_PROPERTIES
    directions = directions if directions is not None else dict(zip(ADMET_PROPERTIES, ADMET_OPTIM_DIRECTIONS))
    if not properties:
        return 0.0
    scores = [
        _normalize_admet_property(prop, admet_preds[prop], directions[prop])
        for prop in properties
    ]
    return sum(scores) / len(scores)


BINDING_SCALE_UM = 10.0
SA_SCORE_MIN = 1.0
SA_SCORE_MAX = 10.0


def _normalize_binding(binding_uM: float) -> float:
    return BINDING_SCALE_UM / (binding_uM + BINDING_SCALE_UM)


def _normalize_sa_score(sa_score: float) -> float:
    clipped = min(max(sa_score, SA_SCORE_MIN), SA_SCORE_MAX)
    return (SA_SCORE_MAX - clipped) / (SA_SCORE_MAX - SA_SCORE_MIN)

SELECTIVITY_SCALE = 10.0


def _normalize_selectivity(selectivity: float) -> float:
    if selectivity is None or selectivity < 0:
        return 0.0
    return selectivity / (selectivity + SELECTIVITY_SCALE)


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
    admet_properties: Optional[Sequence[str]] = None,
    admet_directions: Optional[Dict[str, int]] = None,
) -> dict:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {
            "reward": 0.0, "admet": 0.0, "binding_uM": None, "sa_score": None, "selectivity": None,
            "reward_vector": [0.0, 0.0, 0.0, 0.0],
        }

    admet_model = admet_model if admet_model is not None else ADMETModel(device)
    binding_model = binding_model if binding_model is not None else Plapt(device=str(device))
    sa_model = sa_model if sa_model is not None else SyntheticAccessibility()

    admet_preds = admet_model.predict(smiles)
    admet_reward = compute_admet_reward(admet_preds, admet_properties, admet_directions)

    binding_uM = run_predictions(binding_model, target_seq, [smiles])[0]
    sa_score = sa_model.calculateScore(mol)
    binding_score = _normalize_binding(binding_uM)
    sa_reward = _normalize_sa_score(sa_score)

    if off_target_seq:
        off_target_uM = run_predictions(binding_model, off_target_seq, [smiles])[0]
        selectivity = compare_affinities(binding_uM, off_target_uM)
        selectivity_score = _normalize_selectivity(selectivity)

        scale = selectivity_weight / 3
        reward = (
            (admet_weight - scale) * admet_reward
            + (binding_weight - scale) * binding_score
            + (synthetic_weight - scale) * sa_reward
            + selectivity_weight * selectivity_score
        )
    else:
        selectivity = None
        reward = (
            admet_weight * admet_reward
            + binding_weight * binding_score
            + synthetic_weight * sa_reward
        )

    return {
        "reward": reward,
        "admet": admet_reward,
        "binding_uM": binding_uM,
        "sa_score": sa_score,
        "selectivity": selectivity,
        "reward_vector": [
            admet_reward,
            binding_score,
            sa_reward,
            selectivity_score if off_target_seq else 0.0,
        ],
    }


@dataclass
class RewardConfig:
    admet_weight: float = 0.3
    binding_weight: float = 0.5
    synthetic_weight: float = 0.2
    selectivity_weight: float = 0.3
    admet_properties: Sequence[str] = field(default_factory=lambda: tuple(ADMET_PROPERTIES))
    admet_directions: Dict[str, int] = field(
        default_factory=lambda: dict(zip(ADMET_PROPERTIES, ADMET_OPTIM_DIRECTIONS))
    )

    def __post_init__(self):
        unknown = set(self.admet_properties) - set(ADMET_PROPERTIES)
        if unknown:
            raise ValueError(f"Unknown ADMET properties: {unknown}")
        defaults = dict(zip(ADMET_PROPERTIES, ADMET_OPTIM_DIRECTIONS))
        self.admet_directions = {
            prop: self.admet_directions.get(prop, defaults[prop]) for prop in self.admet_properties
        }

    def to_compute_reward_kwargs(self) -> dict:
        return dict(
            admet_weight=self.admet_weight,
            binding_weight=self.binding_weight,
            synthetic_weight=self.synthetic_weight,
            selectivity_weight=self.selectivity_weight,
            admet_properties=self.admet_properties,
            admet_directions=self.admet_directions,
        )
