import math

import pytest
from rdkit import Chem

from studio.data.loader import (DATA_DIR, dynamicbind_ligand_csv, get_target,
                                list_targets, structure_path)
from studio.schema import TargetInfo

IDS = [t["id"] for t in list_targets()]


def test_index_has_at_least_four():
    assert len(IDS) >= 4


@pytest.mark.parametrize("tid", IDS)
def test_target_valid(tid):
    t = get_target(tid)
    assert isinstance(t, TargetInfo) and t.id == tid
    assert 3 <= len(t.drugs) <= 5
    assert t.pocket_residues and t.uniprot and t.sequence.isalpha()


@pytest.mark.parametrize("tid", IDS)
def test_drugs_parse(tid):
    for d in get_target(tid).drugs:
        m = Chem.MolFromSmiles(d.smiles)
        assert m is not None, d.name
        assert 20 <= m.GetNumHeavyAtoms() <= 50
        assert d.chembl_id and d.chembl_id.startswith("CHEMBL") and d.source


@pytest.mark.parametrize("tid", IDS)
def test_structure_matches_sequence(tid):
    t = get_target(tid)
    p = structure_path(tid)
    assert p.is_file()
    res, resnums = set(), set()
    for l in p.read_text().splitlines():
        if l.startswith("ATOM") and l[21] == t.chain:
            res.add((l[21], l[22:27]))
            resnums.add(int(l[22:26]))
        assert not l.startswith("HETATM")
    assert res, "chain missing"
    # resolved residues vs sequence: within 5% (unresolved loops are the gap)
    assert abs(len(res) - len(t.sequence)) / len(t.sequence) <= 0.05
    assert set(t.pocket_residues) <= resnums


@pytest.mark.parametrize("tid", IDS)
def test_dynamicbind_csv(tid):
    lines = dynamicbind_ligand_csv(tid).read_text().splitlines()
    assert lines[0] == "ligand"
    assert lines[1:] == [d.smiles for d in get_target(tid).drugs]


def test_unknown():
    with pytest.raises(KeyError):
        get_target("nope")
