"""GA building blocks (DEAP): individuals, operators registered on a Toolbox, selection, niching.

Design (simplest robust option):
  * parents: size-k tournament on composite score (local ``random.Random``; DEAP's own selectors use the
    global RNG, which would break per-run determinism)
  * survival: (mu + lambda) elitist truncation with Tanimoto niching. Candidates are taken best-first; a
    candidate whose Morgan Tanimoto to an already-kept member is >= ``niche_threshold`` is deferred and
    only used to fill remaining slots. The best molecule is therefore always kept (elitism).

Transition tuples handed to ``Policy.update`` are
``(parent_smiles, edit_op, site, child_smiles, reward_delta)`` where ``reward_delta`` is the child's
composite minus the parent's composite.
"""
from __future__ import annotations

import itertools
import math
import random
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from deap import base
from rdkit import Chem, DataStructs
from rdkit.Chem import BRICS, rdFingerprintGenerator

from studio.contracts import EditOp, Policy, Registry
from studio.schema import EditRecord, RunConfig, ScoreResult

MAX_HEAVY_ATOMS = 60  # same limit as /api/molecules/validate


class FitnessMax(base.Fitness):
    weights = (1.0,)


@dataclass(eq=False)
class Individual:
    smiles: str
    locked: Tuple[int, ...] = ()  # indices of the seed's locked atoms inside *this* molecule
    score: Optional[ScoreResult] = None
    cid: Optional[str] = None  # candidate id (None for the seed)
    parent: Optional["Individual"] = None  # primary parent (lineage / edit path)
    edit: Optional[EditRecord] = None
    edit_op: Optional[EditOp] = None
    generation: int = 0
    fitness: FitnessMax = field(default_factory=FitnessMax)

    @property
    def composite(self) -> float:
        return self.score.composite if self.score is not None else float("-inf")

    def set_score(self, score: ScoreResult) -> None:
        self.score = score
        self.fitness.values = (score.composite,)


@dataclass
class Proposal:
    """A valid, lock-respecting, not-yet-scored child."""
    smiles: str
    parents: List[Individual]
    edit: EditRecord
    edit_op: Optional[EditOp]
    site: Tuple[int, ...]
    locked: Tuple[int, ...]


def canonical_valid(smiles: str, max_heavy: int = MAX_HEAVY_ATOMS) -> Optional[str]:
    """Canonical SMILES if RDKit sanitizes it to one connected, dummy-free molecule; else None."""
    if not smiles:
        return None
    m = Chem.MolFromSmiles(smiles)
    if m is None or m.GetNumHeavyAtoms() == 0 or m.GetNumHeavyAtoms() > max_heavy:
        return None
    if any(a.GetAtomicNum() == 0 for a in m.GetAtoms()):
        return None
    out = Chem.MolToSmiles(m)
    return None if "." in out else out


# ----------------------------------------------------------------------------- locked atoms
class Locks:
    """Seed atoms that must survive every edit. Verified by MCS tracking + substructure."""

    def __init__(self, seed_smiles: str, locked: Sequence[int],
                 mapper: Callable[[str, str], List[Tuple[int, int]]]):
        self._mapper = mapper
        self.seed_locked = tuple(sorted(set(locked)))
        self._patt = None
        if self.seed_locked:
            seed = Chem.MolFromSmiles(seed_smiles)
            self._patt = Chem.MolFromSmarts(Chem.MolFragmentToSmarts(seed, atomsToUse=list(self.seed_locked)))

    @property
    def active(self) -> bool:
        return bool(self.seed_locked)

    def track(self, parent: Individual, child_smiles: str) -> Optional[Tuple[int, ...]]:
        """Child's locked atom indices, or None when a locked atom (or the locked fragment) was lost."""
        if not self.active:
            return ()
        amap = dict(self._mapper(parent.smiles, child_smiles))
        if any(i not in amap for i in parent.locked):
            return None
        mapped = tuple(amap[i] for i in parent.locked)
        child = Chem.MolFromSmiles(child_smiles)
        want = set(mapped)
        for match in child.GetSubstructMatches(self._patt, uniquify=False, maxMatches=2000):
            if set(match) == want:
                return mapped
        return None


# ----------------------------------------------------------------------------- operators
class Operators:
    def __init__(self, registry: Registry, policy: Policy, config: RunConfig, rng: random.Random,
                 locks: Locks, crossover_rate: float = 0.3):
        self.registry, self.policy, self.config, self.rng, self.locks = registry, policy, config, rng, locks
        self.crossover_rate = crossover_rate
        self._cands: Dict[Tuple[str, Tuple[int, ...]], List[tuple]] = {}
        self._probs: Dict[Tuple[str, Tuple[int, ...]], List[float]] = {}
        self._frags: Dict[str, List[str]] = {}
        self.policy_errors = 0
        self.toolbox = base.Toolbox()  # DEAP registry of the operators the engine loop uses
        self.toolbox.register("select", self.tournament)
        self.toolbox.register("mutate", self.mutate)
        self.toolbox.register("mate", self.crossover)

    # -- selection
    def tournament(self, pop: Sequence[Individual], k: int = 3) -> Individual:
        picks = self.rng.sample(list(pop), min(k, len(pop)))
        return max(picks, key=lambda i: (i.composite, i.smiles))

    def new_generation_caches(self) -> None:
        """Policy probabilities go stale after Policy.update; edit applicability does not."""
        self._probs.clear()

    # -- mutation
    def _candidates(self, ind: Individual) -> List[tuple]:
        key = (ind.smiles, ind.locked)
        if key not in self._cands:
            self._cands[key] = list(self.registry.applicable(ind.smiles, ind.locked))
        return self._cands[key]

    def _policy_probs(self, ind: Individual, cands: List[tuple]) -> Optional[List[float]]:
        key = (ind.smiles, ind.locked)
        if key not in self._probs:
            try:
                scores = list(self.policy.rank(ind.smiles, cands))
                mx = max(scores)
                w = [math.exp(s - mx) for s in scores]
                z = sum(w)
                self._probs[key] = [x / z for x in w]
            except Exception:
                self.policy_errors += 1
                self._probs[key] = []
        return self._probs[key] or None

    def _sample(self, probs: Sequence[float]) -> int:
        r, acc = self.rng.random(), 0.0
        for i, p in enumerate(probs):
            acc += p
            if r < acc:
                return i
        return len(probs) - 1

    def mutate(self, ind: Individual) -> Optional[Proposal]:
        cands = self._candidates(ind)
        if not cands:
            return None
        chosen_by, probs = "random", None
        if self.rng.random() < self.config.policy_guidance:
            probs = self._policy_probs(ind, cands)
            if probs:
                chosen_by = "policy"
        idx = self._sample(probs) if chosen_by == "policy" else self.rng.randrange(len(cands))
        op, site = cands[idx]
        products = op.apply(ind.smiles, site)
        valid = [s for s in (canonical_valid(p) for p in products) if s and s != ind.smiles]
        if not valid:
            return None
        child = valid[self.rng.randrange(len(valid))]
        locked = self.locks.track(ind, child)
        if locked is None:
            return None
        rec = EditRecord(edit_id=op.id, name=op.name, category=op.category, site=list(site),
                         chosen_by=chosen_by, parents=[ind.smiles])
        return Proposal(child, [ind], rec, op, tuple(site), locked)

    # -- crossover (BRICS)
    def _brics(self, smiles: str) -> List[str]:
        if smiles not in self._frags:
            self._frags[smiles] = sorted(BRICS.BRICSDecompose(Chem.MolFromSmiles(smiles)))
        return self._frags[smiles]

    def crossover(self, a: Individual, b: Individual, max_products: int = 12) -> Optional[Proposal]:
        fa, fb = self._brics(a.smiles), self._brics(b.smiles)
        if a.smiles == b.smiles or len(fa) < 2 and len(fb) < 2:
            return None
        pool = sorted(set(fa) | set(fb))
        # BRICSBuild enumerates exponentially in pool size and depth (minutes on drug-sized parents)
        pool = sorted(self.rng.sample(pool, min(len(pool), 4)))
        seed_frag = self.rng.choice(fa)
        try:
            frags = [Chem.MolFromSmiles(s) for s in sorted(set(pool) | {seed_frag})]
            gen = BRICS.BRICSBuild(frags, onlyCompleteMols=True, seeds=[Chem.MolFromSmiles(seed_frag)],
                                   scrambleReagents=False, maxDepth=2)
            raw = []
            for m in itertools.islice(gen, max_products):
                try:
                    Chem.SanitizeMol(m)
                    raw.append(Chem.MolToSmiles(m))
                except Exception:
                    continue
        except Exception:
            return None
        valid = [s for s in (canonical_valid(p) for p in raw) if s and s not in (a.smiles, b.smiles)]
        if not valid:
            return None
        child = sorted(set(valid))[self.rng.randrange(len(set(valid)))]
        locked = self.locks.track(a, child)
        if locked is None:
            return None
        rec = EditRecord(edit_id="brics_crossover", name="BRICS crossover", category="crossover",
                         site=[], chosen_by="crossover", parents=[a.smiles, b.smiles])
        return Proposal(child, [a, b], rec, None, (), locked)

    def propose(self, pop: Sequence[Individual]) -> Optional[Proposal]:
        """One attempt at a child: crossover with probability ``crossover_rate``, otherwise mutation."""
        if self.config.use_crossover and len({i.smiles for i in pop}) >= 2 and self.rng.random() < self.crossover_rate:
            a = self.toolbox.select(pop)
            others = [i for i in pop if i.smiles != a.smiles]
            b = self.toolbox.select(others)
            return self.toolbox.mate(a, b)
        return self.toolbox.mutate(self.toolbox.select(pop))


# ----------------------------------------------------------------------------- survival + niching
_FPGEN = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)


class Niching:
    def __init__(self, threshold: float = 0.85):
        self.threshold = threshold
        self._fp: Dict[str, object] = {}

    def fp(self, smiles: str):
        if smiles not in self._fp:
            self._fp[smiles] = _FPGEN.GetFingerprint(Chem.MolFromSmiles(smiles))
        return self._fp[smiles]

    def survive(self, pool: Sequence[Individual], size: int) -> List[Individual]:
        ranked = sorted(pool, key=lambda i: (-i.composite, i.smiles))
        kept: List[Individual] = []
        deferred: List[Individual] = []
        kept_fps = []
        for ind in ranked:
            if len(kept) >= size:
                break
            f = self.fp(ind.smiles)
            if kept_fps and max(DataStructs.BulkTanimotoSimilarity(f, kept_fps)) >= self.threshold:
                deferred.append(ind)
                continue
            kept.append(ind)
            kept_fps.append(f)
        for ind in deferred:
            if len(kept) >= size:
                break
            kept.append(ind)
        return sorted(kept, key=lambda i: (-i.composite, i.smiles))
