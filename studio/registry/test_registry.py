import pytest
from rdkit import Chem

from studio.contracts import EditOp, Registry
from studio.registry import get_registry

SEED = "CCOC(=O)c1ccc(cc1)[N+](=O)[O-]"
reg = get_registry()


def test_protocols_and_count():
    assert isinstance(reg, Registry)
    assert len(reg.all()) == 83
    assert all(isinstance(o, EditOp) and o.rationale and o.tags for o in reg.all())
    assert all(not o.rationale_reviewed for o in reg.all())  # until the human owner signs off


def test_seed_applicable_and_valid():
    app = reg.applicable(SEED)
    assert app
    for op, site in app[:200]:
        for p in op.apply(SEED, site):
            m = Chem.MolFromSmiles(p)
            assert m is not None and Chem.MolToSmiles(m) == p and p != Chem.CanonSmiles(SEED)


def test_locked_atoms_excluded():
    locked = [0, 1, 2]
    for _, site in reg.applicable(SEED, locked):
        assert not set(site) & set(locked)


@pytest.mark.parametrize("bad", ["", "not a smiles", "[Xx]", None, "C.C"])
def test_never_raises_on_bad_input(bad):
    for op in reg.all():
        assert op.sites(bad) in ([], op.sites(bad))
        assert op.apply(bad, [0]) == [] or bad == "C.C"
        assert op.apply(SEED, [999]) == []
        assert op.apply(SEED, []) == []


def test_deterministic():
    for op, site in reg.applicable(SEED)[:100]:
        assert op.apply(SEED, site) == op.apply(SEED, site)


def test_demo_safe_filter(monkeypatch):
    from studio.registry import get_registry
    r = get_registry()
    smi = "CCOC(=O)c1ccc(cc1)[N+](=O)[O-]"
    monkeypatch.delenv("STUDIO_ALL_EDITS", raising=False)
    safe = r.applicable(smi)
    assert safe and all(op.demo_safe for op, _ in safe)
    assert not any(op.id in ("atom_sub_s", "atom_add_br") for op, _ in safe)
    allx = r.applicable(smi, demo_safe_only=False)
    assert len(allx) > len(safe)
    monkeypatch.setenv("STUDIO_ALL_EDITS", "1")
    assert len(r.applicable(smi)) == len(allx)
    assert sum(op.demo_safe for op in r.all()) < len(r.all())
