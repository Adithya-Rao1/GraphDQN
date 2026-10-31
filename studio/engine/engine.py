"""Engine: the GA + policy loop that yields schema ``Event``s.

Per generation: propose ``population_size`` valid, lock-respecting, unseen children -> score them (batch) ->
(mu + lambda) survival with Tanimoto niching -> emit ``candidate`` events (every accepted child plus the best
``max_rejected_events`` rejected ones; the rest are counted but not sent, to keep the SSE stream small)
-> ``generation`` -> online ``Policy.update`` -> ``policy_metrics``.

Everything random goes through one ``random.Random(config.rng_seed)``, so a run is reproducible given
deterministic collaborators.
"""
from __future__ import annotations

import random
import time
from typing import Dict, Iterator, List, Optional, Tuple

from rdkit import Chem

from studio.contracts import Embedder, Explainer, Policy, Registry, Scorer
from studio.engine import geometry
from studio.engine.ga import Individual, Locks, Niching, Operators, Proposal, canonical_valid
from studio.schema import (
    Candidate, ErrorPayload, Event, GenerationPayload, PolicyMetrics, PopulationMember,
    RunConfig, RunStartPayload, ScoreResult, StoryCard,
)

ATTEMPT_FACTOR = 8  # proposal attempts per wanted child before a generation gives up


class _Emitter:
    def __init__(self, run_id: str):
        self.run_id, self.seq = run_id, 0

    def __call__(self, type_: str, payload) -> Event:
        ev = Event(type=type_, run_id=self.run_id, seq=self.seq, payload=payload.model_dump(mode="json"))
        self.seq += 1
        return ev


class Engine:
    def __init__(self, registry: Registry, scorer: Scorer, explainer: Explainer,
                 embedder: Optional[Embedder], policy: Policy, *,
                 crossover_rate: float = 0.3, niche_threshold: float = 0.85, max_rejected_events: int = 5):
        self.registry, self.scorer, self.explainer = registry, scorer, explainer
        self.embedder, self.policy = embedder, policy
        self.crossover_rate, self.niche_threshold = crossover_rate, niche_threshold
        self.max_rejected_events = max_rejected_events
        self.stats: Dict[str, float] = {}  # filled by run(): children, seconds, children_per_second

    # ------------------------------------------------------------------ public
    def run(self, run_id: str, config: RunConfig) -> Iterator[Event]:
        emit = _Emitter(run_id)
        try:
            yield from self._run(run_id, config, emit)
        except Exception as exc:  # surface as a terminal error event instead of killing the SSE stream
            yield emit("error", ErrorPayload(message=f"{type(exc).__name__}: {exc}", recoverable=False))

    # ------------------------------------------------------------------ internals
    @staticmethod
    def _canonical_seed(config: RunConfig) -> Tuple[str, Tuple[int, ...]]:
        """Canonical seed SMILES and locked indices renumbered to it (config indices follow MolFromSmiles(seed))."""
        mol = Chem.MolFromSmiles(config.seed_smiles)
        if mol is None:
            raise ValueError("seed_smiles is not a valid molecule")
        canon = canonical_valid(config.seed_smiles)
        if canon is None:
            raise ValueError("seed must be one connected molecule of at most 60 heavy atoms")
        n = mol.GetNumAtoms()
        bad = [i for i in config.locked_atoms if not 0 <= i < n]
        if bad:
            raise ValueError(f"locked_atoms out of range for the seed: {bad}")
        Chem.MolToSmiles(mol)
        order = list(mol.GetPropsAsDict(includePrivate=True, includeComputed=True)["_smilesAtomOutputOrder"])
        return canon, tuple(sorted(order.index(i) for i in set(config.locked_atoms)))

    def _run(self, run_id: str, config: RunConfig, emit: _Emitter) -> Iterator[Event]:
        t0 = time.perf_counter()
        n_children = 0
        rng = random.Random(config.rng_seed)
        seed_smiles, seed_locked = self._canonical_seed(config)

        geo: Dict[str, Chem.Mol] = {}
        scores: Dict[str, ScoreResult] = {}
        amaps: Dict[Tuple[str, str], List[Tuple[int, int]]] = {}

        def mapper(p: str, c: str) -> List[Tuple[int, int]]:
            if (p, c) not in amaps:
                amaps[(p, c)] = geometry.atom_map(p, c)
            return amaps[(p, c)]

        seed_block, geo[seed_smiles] = geometry.embed_3d(seed_smiles)
        seed_score = scores[seed_smiles] = self.scorer.score(seed_smiles, config)
        yield emit("run_start", RunStartPayload(config=config, seed_molblock=seed_block,
                                                seed_score=seed_score, source="live"))

        seed = Individual(smiles=seed_smiles, locked=seed_locked, generation=0)
        seed.set_score(seed_score)
        locks = Locks(seed_smiles, seed_locked, mapper)
        ops = Operators(self.registry, self.policy, config, rng, locks, self.crossover_rate)
        niching = Niching(self.niche_threshold)
        pop: List[Individual] = [seed]
        seen = {seed_smiles}

        for gen in range(1, config.generations + 1):
            # 1. propose valid, lock-respecting, unseen, embeddable children
            props: List[Proposal] = []
            attempts, budget = 0, ATTEMPT_FACTOR * config.population_size
            while len(props) < config.population_size and attempts < budget:
                attempts += 1
                p = ops.propose(pop)
                if p is None or p.smiles in seen:
                    continue
                seen.add(p.smiles)
                mol = geometry.embed_raw(p.smiles)
                if mol is None:
                    continue
                geo[p.smiles] = mol
                props.append(p)
            n_children += len(props)

            # 2. score as one batch (cache: identical SMILES are scored once per run)
            todo = [p.smiles for p in props if p.smiles not in scores]
            if todo:
                for s, sc in zip(todo, self.scorer.score_many(todo, config)):
                    scores[s] = sc
            kids: List[Individual] = []
            for i, p in enumerate(props):
                kid = Individual(smiles=p.smiles, locked=p.locked, cid=f"{run_id}-g{gen}-{i}", parent=p.parents[0],
                                 edit=p.edit, edit_op=p.edit_op, generation=gen)
                kid.set_score(scores[p.smiles])
                kids.append(kid)

            # 3. survival
            new_pop = niching.survive(pop + kids, config.population_size)
            kept = {id(k) for k in new_pop}
            accepted = [k for k in kids if id(k) in kept]
            rejected = sorted((k for k in kids if id(k) not in kept), key=lambda k: (-k.composite, k.cid))
            to_emit = set(id(k) for k in accepted) | {id(k) for k in rejected[: self.max_rejected_events]}

            # 4. candidate events (3D aligned to the parent, atom map, explanation)
            for kid in kids:
                if id(kid) in to_emit:
                    yield emit("candidate", self._candidate(kid, id(kid) in kept, geo, mapper))
            pop = new_pop

            # 5. generation summary
            members = [PopulationMember(smiles=i.smiles, composite=i.composite, candidate_id=i.cid) for i in pop]
            map_xy, map_method = None, None
            if self.embedder is not None:
                try:
                    map_xy = [(float(x), float(y)) for x, y in
                              self.embedder.map_xy([i.smiles for i in pop], anchor=seed_smiles)]
                    map_method = self.embedder.method
                except Exception:
                    map_xy, map_method = None, None  # the map is optional; never fail the run for it
            yield emit("generation", GenerationPayload(
                generation=gen, best=members[0], mean_composite=sum(m.composite for m in members) / len(members),
                population=members, map_xy=map_xy, map_method=map_method))

            # 6. online learning + policy metrics
            yield emit("policy_metrics", self._policy_metrics(gen, kids, accepted, ops))
            ops.new_generation_caches()

        best = pop[0]
        self.stats = {"children": float(n_children), "seconds": time.perf_counter() - t0}
        self.stats["children_per_second"] = n_children / self.stats["seconds"] if self.stats["seconds"] else 0.0
        yield emit("run_end", self._story(config, seed_smiles, seed_score, best))

    def _candidate(self, kid: Individual, accepted: bool, geo: Dict[str, Chem.Mol], mapper) -> Candidate:
        parent = kid.parent
        pmol, cmol = geo[parent.smiles], geo[kid.smiles]
        amap = mapper(parent.smiles, kid.smiles)
        geometry.align_to(cmol, pmol, amap)
        edit = kid.edit
        if edit.category == "crossover":  # site = parent atoms that were not carried over
            mapped = {p for p, _ in amap}
            edit = edit.model_copy(update={"site": [i for i in range(pmol.GetNumAtoms()) if i not in mapped]})
        expl = self.explainer.explain(parent.smiles, kid.smiles, edit, parent.score, kid.score, kid.edit_op)
        return Candidate(
            id=kid.cid, generation=kid.generation, smiles=kid.smiles, parent_smiles=parent.smiles,
            molblock=Chem.MolToMolBlock(cmol), parent_molblock=Chem.MolToMolBlock(pmol),
            atom_map=[(int(p), int(c)) for p, c in amap], edit=edit, score=kid.score,
            explanation=expl, accepted=accepted)

    def _policy_metrics(self, gen: int, kids: List[Individual], accepted: List[Individual], ops: Operators) -> PolicyMetrics:
        muts = [k for k in kids if k.edit.chosen_by in ("policy", "random")]
        reward = lambda k: k.composite - k.parent.composite  # noqa: E731
        pol = [reward(k) for k in muts if k.edit.chosen_by == "policy"]
        rnd = [reward(k) for k in muts if k.edit.chosen_by == "random"]
        notes = []
        if not kids:
            notes.append("no valid new children this generation")
        transitions = [(k.parent.smiles, k.edit_op, tuple(k.edit.site), k.smiles, reward(k))
                       for k in accepted if k.edit_op is not None]
        loss = None
        if transitions:
            try:
                loss = self.policy.update(transitions)
            except Exception as exc:
                notes.append(f"policy update failed: {type(exc).__name__}")
        if ops.policy_errors:
            notes.append(f"policy.rank failed {ops.policy_errors}x, fell back to random")
        pnote = getattr(self.policy, "note", None)
        if pnote:
            notes.append(str(pnote))
        return PolicyMetrics(
            generation=gen, policy_choice_rate=len(pol) / len(muts) if muts else 0.0,
            mean_reward_policy_edits=sum(pol) / len(pol) if pol else None,
            mean_reward_random_edits=sum(rnd) / len(rnd) if rnd else None,
            loss=float(loss) if loss is not None else None, note="; ".join(notes) or None)

    @staticmethod
    def _story(config: RunConfig, seed_smiles: str, seed_score: ScoreResult, best: Individual) -> StoryCard:
        path: List[str] = []
        node = best
        while node is not None and node.cid is not None:
            path.append(node.cid)
            node = node.parent
        path.reverse()
        a, b = seed_score.descriptors, best.score.descriptors
        parts = []
        if b.alerts < a.alerts:
            parts.append(f"Removed {a.alerts - b.alerts} alert{'s' if a.alerts - b.alerts != 1 else ''}")
        elif b.alerts > a.alerts:
            parts.append(f"Added {b.alerts - a.alerts} alert{'s' if b.alerts - a.alerts != 1 else ''}")
        parts.append(f"QED {a.qed:.2f} -> {b.qed:.2f}")
        parts.append(f"composite {seed_score.composite:.2f} -> {best.score.composite:.2f}")
        headline = ", ".join(parts) + (f" in {len(path)} edit{'s' if len(path) != 1 else ''}" if path
                                       else " (no improving edit found)")
        da, db = a.model_dump(), b.model_dump()
        table = {k: (float(da[k]), float(db[k])) for k in da if da[k] is not None and db[k] is not None}
        return StoryCard(seed_smiles=seed_smiles, final_smiles=best.smiles, seed_score=seed_score,
                         final_score=best.score, edit_path=path, headline=headline, descriptor_table=table)
