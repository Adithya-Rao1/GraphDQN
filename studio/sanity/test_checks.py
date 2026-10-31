import pytest

from studio.sanity import check_child

NITRO = "CCOC(=O)c1ccc(cc1)[N+](=O)[O-]"

# (rule, parent, bad child, expected reason prefix)
NEGATIVE = [
    ("sanitize_failed", NITRO, "CC1=C1", "sanitize_failed"),
    ("fragments", NITRO, "CCOC(=O)c1ccc(cc1)[N+](=O)[O-].C", "fragments"),
    ("radical", "CC", "[CH2]C", "radical"),
    ("nitroso case from the integrator", NITRO, "CNOC(=O)c1ccc([N+]=O)cc1", "radical"),
    ("net_charge_changed", "CC", "C[N+](C)(C)C", "net_charge_changed"),
    ("unrecognised_charge", "CC[N+](C)(C)C", "C[N+](C)(C)CC[N+](C)(C)CC[O-]", "unrecognised_charge"),
    ("nitroso_cation", "CC[N+](C)(C)C", "CC[N+](C)=O", "bad_pattern:nitroso_cation"),
    ("peroxide", NITRO, "CCOOC(=O)c1ccc(cc1)[N+](=O)[O-]", "bad_pattern:peroxide"),
    ("n_n_n_chain", "CC", "CNNN", "bad_pattern:n_n_n_chain"),
    ("o_acyl_hydroxylamine", "CC(=O)OC", "CC(=O)ON(C)C", "bad_pattern:o_acyl_hydroxylamine"),
    ("n_halogen", "CCN", "CCNCl", "bad_pattern:n_halogen"),
    ("acyl_halide", "CC(=O)O", "CC(=O)Cl", "bad_pattern:acyl_halide"),
    ("s_s_s", "CSSC", "CSSSC", "bad_pattern:s_s_s"),
    ("ring3_two_heteroatoms", "CC", "C1NN1", "bad_pattern:ring3_two_heteroatoms"),
    ("small_ring_triple", "CC", "C1CCCC#CC1", "bad_pattern:small_ring_triple"),
    ("cyclopropene", "CC", "CC1=CC1", "bad_pattern:cyclopropene"),
    ("gem_triheteroatom", "COC", "COC(OC)OC", "bad_pattern:gem_triheteroatom"),
]

POSITIVE = [
    ("nitro -> nitrile", NITRO, "CCOC(=O)c1ccc(C#N)cc1"),
    ("ester -> amide", NITRO, "CCNC(=O)c1ccc(cc1)[N+](=O)[O-]"),
    ("hydroxamic acid", "CC(=O)O", "CC(=O)NO"),
    ("hydroxamic acid kept", "CC(=O)NO", "CCC(=O)NO"),
    ("N-oxide", "c1ccncc1", "c1cc[n+]([O-])cc1"),
    ("disulfide", "CCSSC", "CCSSCC"),
    ("hydrazide", "CC(=O)N", "CC(=O)NN"),
    ("sulfonamide", "c1ccccc1", "c1ccccc1S(N)(=O)=O"),
    ("sulfoxide", "CSC", "CS(C)=O"),
    ("azide", "CC", "CCN=[N+]=[N-]"),
    ("ammonium/carboxylate zwitterion", "NCC(=O)O", "[NH3+]CC(=O)[O-]"),
    ("cyclopropane", "CC", "C1CC1"),
    ("epoxide", "CC", "C1CO1"),
    ("parent already has the motif", "CC1=CC1", "CCC1=CC1"),
    ("no parent given", None, "CCO"),
]


@pytest.mark.parametrize("rule,parent,child,reason", NEGATIVE)
def test_rejects(rule, parent, child, reason):
    ok, why = check_child(parent, child)
    assert not ok and why == reason, (rule, why)


@pytest.mark.parametrize("rule,parent,child", POSITIVE)
def test_allows(rule, parent, child):
    ok, why = check_child(parent, child)
    assert ok and why == "ok", (rule, why)


def test_no_scorer_dependency():
    import sys
    import studio.sanity.checks as m
    assert "studio.scoring" not in {getattr(v, "__module__", "") for v in vars(m).values()}
