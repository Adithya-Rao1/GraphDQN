"""ProfileScorer: pure wrapper that applies a saved domain profile on top of any Scorer.

* profile.objectives override/extend RunConfig.objectives before the base scorer runs.
* every avoid-SMARTS that matches adds AlertHit(family="USER") (first match's atoms) and
  multiplies the composite by USER_PENALTY per matching pattern (documented, deliberately simple).
* descriptors.alerts is increased by the number of USER hits; components["user_avoid"] is added.

Server resolution (for the integrator): when RunConfig.profile_id is set and the user owns it,
    from studio.accounts.profile_scorer import ProfileScorer, load_profile_for
    prof = load_profile_for(request, profile_id)      # None if absent / not owned (treat as 404)
    scorer = ProfileScorer(base_scorer, prof) if prof else base_scorer
and use `scorer` for that run only. `profile_id` is not in schema.py yet (report to integrator).
"""
from __future__ import annotations

from typing import List, Optional, Sequence

from rdkit import Chem

from studio.accounts.profiles import ProfileBody
from studio.schema import AlertHit, RunConfig, ScoreResult

USER_PENALTY = 0.8


class ProfileScorer:
    def __init__(self, base, profile) -> None:
        self.base = base
        self.profile = profile if isinstance(profile, ProfileBody) else ProfileBody.model_validate(profile)
        self._patterns = [(s, Chem.MolFromSmarts(s)) for s in self.profile.avoid_smarts]
        self._patterns = [(s, q) for s, q in self._patterns if q is not None]

    def _config(self, config: RunConfig) -> RunConfig:
        if not self.profile.objectives:
            return config
        merged = {**config.objectives, **self.profile.objectives}
        return config.model_copy(update={"objectives": merged})

    def score(self, smiles: str, config: RunConfig) -> ScoreResult:
        res = self.base.score(smiles, self._config(config))
        mol = Chem.MolFromSmiles(smiles)
        hits: List[AlertHit] = []
        if mol is not None:
            for smarts, q in self._patterns:
                m = mol.GetSubstructMatch(q)
                if m:
                    hits.append(AlertHit(name=f"avoid: {smarts}", family="USER", atom_indices=list(m)))
        if not hits:
            return res
        n = len(hits)
        desc = res.descriptors.model_copy(update={"alerts": res.descriptors.alerts + n})
        comps = {**res.components, "user_avoid": max(0.0, 1.0 - 0.2 * n)}
        return res.model_copy(update={
            "descriptors": desc, "components": comps, "alerts": [*res.alerts, *hits],
            "composite": max(0.0, res.composite * (USER_PENALTY ** n)),
        })

    def score_many(self, smiles: Sequence[str], config: RunConfig) -> List[ScoreResult]:
        return [self.score(s, config) for s in smiles]


def load_profile_for(request, profile_id: int) -> Optional[ProfileBody]:
    """Owner-checked profile lookup for the run-creation route (None if absent / not yours)."""
    import json
    from sqlalchemy.orm import Session
    from studio.accounts.db import Profile
    from studio.accounts.routes import current_user

    user = current_user(request)
    if user is None:
        return None
    with Session(request.app.state.accounts.engine) as s:
        row = s.get(Profile, profile_id)
        if row is None or row.user_id != user.id:
            return None
        return ProfileBody.model_validate_json(row.json)
