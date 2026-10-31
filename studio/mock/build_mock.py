"""Build studio/mock/mock_run.json: a hand-scripted run with REAL RDKit descriptors.

This is mock *data* (the edit path and the GA population are scripted, not searched),
used so the server, viewer and panels can be developed before the engine exists.
Every event is tagged source="mock". Regenerate with:  python -m studio.mock.build_mock
"""
from __future__ import annotations

import json
from pathlib import Path

from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, Crippen, Descriptors as D, QED, rdFMCS, rdMolDescriptors
from rdkit.Chem.FilterCatalog import FilterCatalog, FilterCatalogParams

from studio.schema import (
    AlertHit, AtomContributions, Candidate, Descriptors, EditRecord, Explanation,
    GenerationPayload, PolicyMetrics, PopulationMember, RunConfig, RunStartPayload,
    ScoreResult, StoryCard, SCHEMA_VERSION,
)

RUN_ID = "mock-run-1"
SEED = "CCOC(=O)c1ccc(cc1)[N+](=O)[O-]"  # ethyl 4-nitrobenzoate: a nitro alert + a labile ester

# (smiles, edit_id, name, category, rationale)
PATH = [
    ("CCOC(=O)c1ccc(cc1)C#N", "nitro_to_nitrile", "Nitro -> nitrile", "bioisostere",
     "Nitroaromatics are a common mutagenicity alert; a nitrile keeps the electron-withdrawing character."),
    ("CCNC(=O)c1ccc(cc1)C#N", "ester_to_amide", "Ester -> amide", "functional_group",
     "Amides resist esterase hydrolysis, improving metabolic stability."),
    ("CCNC(=O)c1ccc(cc1)C(N)=O", "nitrile_to_primary_amide", "Nitrile -> primary amide", "functional_group",
     "Adds a hydrogen-bond donor and polarity, lowering logP."),
]

_params = FilterCatalogParams()
for cat in (FilterCatalogParams.FilterCatalogs.PAINS, FilterCatalogParams.FilterCatalogs.BRENK):
    _params.AddCatalog(cat)
_CATALOG = FilterCatalog(_params)


def _alerts(m):
    hits = []
    for e in _CATALOG.GetMatches(m):
        atoms = sorted({a for _, a in e.GetFilterMatches(m)[0].atomPairs}) if e.GetFilterMatches(m) else []
        hits.append(AlertHit(name=e.GetDescription(), family="BRENK/PAINS", atom_indices=atoms))
    return hits


def _desc(m, alerts) -> Descriptors:
    lip = int(D.MolWt(m) > 500) + int(Crippen.MolLogP(m) > 5) + int(rdMolDescriptors.CalcNumHBD(m) > 5) + int(rdMolDescriptors.CalcNumHBA(m) > 10)
    return Descriptors(
        qed=QED.qed(m), sa=None, logp=Crippen.MolLogP(m), tpsa=rdMolDescriptors.CalcTPSA(m),
        mw=D.MolWt(m), hbd=rdMolDescriptors.CalcNumHBD(m), hba=rdMolDescriptors.CalcNumHBA(m),
        rotb=rdMolDescriptors.CalcNumRotatableBonds(m), fsp3=rdMolDescriptors.CalcFractionCSP3(m),
        heavy_atoms=m.GetNumHeavyAtoms(), lipinski_violations=lip, alerts=len(alerts),
    )


def score(smiles: str) -> ScoreResult:
    m = Chem.MolFromSmiles(smiles)
    al = _alerts(m)
    d = _desc(m, al)
    comps = {
        "qed": d.qed,
        "logp": max(0.0, 1 - abs(d.logp - 2.5) / 3.5),
        "tpsa": max(0.0, 1 - abs(d.tpsa - 75) / 75),
        "alerts": 1.0 if d.alerts == 0 else max(0.0, 1 - 0.34 * d.alerts),
    }
    return ScoreResult(descriptors=d, components=comps, composite=sum(comps.values()) / len(comps), alerts=al)


def molblock_3d(smiles: str, ref=None) -> tuple[str, Chem.Mol]:
    m = Chem.AddHs(Chem.MolFromSmiles(smiles))
    AllChem.EmbedMolecule(m, randomSeed=7)
    AllChem.MMFFOptimizeMolecule(m)
    m = Chem.RemoveHs(m)
    if ref is not None:  # rigid-align on the MCS so the morph is smooth
        mcs = rdFMCS.FindMCS([ref, m], timeout=2)
        patt = Chem.MolFromSmarts(mcs.smartsString)
        a, b = ref.GetSubstructMatch(patt), m.GetSubstructMatch(patt)
        if a and b:
            AllChem.AlignMol(m, ref, atomMap=list(zip(b, a)))
    return Chem.MolToMolBlock(m), m


def atom_map(parent, child):
    mcs = rdFMCS.FindMCS([parent, child], timeout=2)
    patt = Chem.MolFromSmarts(mcs.smartsString)
    a, b = parent.GetSubstructMatch(patt), child.GetSubstructMatch(patt)
    return list(zip(a, b)) if a and b else []


def contributions(m) -> AtomContributions:
    return AtomContributions(
        crippen_logp=[c[0] for c in rdMolDescriptors._CalcCrippenContribs(m)],
        tpsa=list(rdMolDescriptors._CalcTPSAContribs(m)),
        gasteiger=[(AllChem.ComputeGasteigerCharges(m) or 0) or float(a.GetProp("_GasteigerCharge")) for a in m.GetAtoms()],
    )


def build() -> list[dict]:
    cfg = RunConfig(seed_smiles=SEED, generations=len(PATH), population_size=4)
    events, seq = [], 0

    def emit(t, payload):
        nonlocal seq
        events.append({"type": t, "run_id": RUN_ID, "seq": seq, "payload": payload.model_dump(mode="json")})
        seq += 1

    seed_block, seed_mol = molblock_3d(SEED)
    seed_score = score(SEED)
    emit("run_start", RunStartPayload(config=cfg, seed_molblock=seed_block, seed_score=seed_score, source="mock"))

    parent_smiles, parent_block, parent_mol, parent_score = SEED, seed_block, seed_mol, seed_score
    path_ids = []
    for g, (smi, eid, name, cat, why) in enumerate(PATH, start=1):
        block, mol = molblock_3d(smi, ref=parent_mol)
        sc = score(smi)
        deltas = {k: getattr(sc.descriptors, k) - getattr(parent_score.descriptors, k)
                  for k in ("qed", "logp", "tpsa", "mw", "hbd", "hba", "alerts")}
        expl = Explanation(
            descriptor_deltas=deltas,
            alerts_added=[a for a in sc.alerts if a.name not in {x.name for x in parent_score.alerts}],
            alerts_removed=[a for a in parent_score.alerts if a.name not in {x.name for x in sc.alerts}],
            atom_contributions=contributions(mol), rationale=why,
            tags=(["removes_alert"] if deltas["alerts"] < 0 else []) + (["lowers_logp"] if deltas["logp"] < 0 else []),
        )
        cid = f"mock-g{g}-0"
        path_ids.append(cid)
        emit("candidate", Candidate(
            id=cid, generation=g, smiles=smi, parent_smiles=parent_smiles, molblock=block,
            parent_molblock=parent_block, atom_map=atom_map(parent_mol, mol),
            edit=EditRecord(edit_id=eid, name=name, category=cat, site=[], chosen_by="policy", parents=[parent_smiles]),
            score=sc, explanation=expl, accepted=True))
        pop = [PopulationMember(smiles=smi, composite=sc.composite, candidate_id=cid),
               PopulationMember(smiles=parent_smiles, composite=parent_score.composite)]
        emit("generation", GenerationPayload(
            generation=g, best=pop[0], mean_composite=sum(p.composite for p in pop) / len(pop),
            population=pop, map_xy=[(float(g), sc.composite), (float(g - 1), parent_score.composite)], map_method="morgan_pca"))
        emit("policy_metrics", PolicyMetrics(generation=g, policy_choice_rate=0.5, note="mock data"))
        parent_smiles, parent_block, parent_mol, parent_score = smi, block, mol, sc

    table = {k: (getattr(seed_score.descriptors, k), getattr(parent_score.descriptors, k))
             for k in ("qed", "logp", "tpsa", "mw", "alerts")}
    emit("run_end", StoryCard(
        seed_smiles=SEED, final_smiles=parent_smiles, seed_score=seed_score, final_score=parent_score, edit_path=path_ids,
        headline=f"Removed {seed_score.descriptors.alerts - parent_score.descriptors.alerts} alerts, "
                 f"QED {seed_score.descriptors.qed:.2f} -> {parent_score.descriptors.qed:.2f} in {len(PATH)} edits (MOCK DATA)",
        descriptor_table=table))
    return events


if __name__ == "__main__":
    out = Path(__file__).with_name("mock_run.json")
    ev = build()
    out.write_text(json.dumps(ev))
    print(f"wrote {out} ({len(ev)} events, schema {SCHEMA_VERSION})")
